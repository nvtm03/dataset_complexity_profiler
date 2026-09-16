"""CLI: ``dcp analyze``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from pandas.errors import ParserError

from dataset_complexity_profiler.defaults import DEFAULT_EMBEDDER_NAME, DEFAULT_QUALITY_THRESHOLD

_CLI_USER_ERRORS = (
    ImportError,
    TypeError,
    ValueError,
    FileNotFoundError,
    KeyError,
    OSError,
    ParserError,
)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _nonempty_text(value: object) -> bool:
    return not _is_missing(value) and str(value).strip() != ""


def _read_csv_texts_labels(
    path: Path,
    text_col: str,
    label_col: str,
    limit: Optional[int],
    text_col_b: Optional[str] = None,
) -> Tuple[List[str], List[object], Optional[List[str]]]:
    import pandas as pd

    if limit is not None and limit < 1:
        raise ValueError(f"Limit must be >= 1, got {limit}")

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if text_col not in columns or label_col not in columns:
        raise ValueError(
            f"CSV must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {columns}"
        )
    if text_col_b and text_col_b not in columns:
        raise ValueError(f"CSV missing second text column '{text_col_b}'")

    usecols = [text_col, label_col]
    if text_col_b:
        usecols.append(text_col_b)

    def _keep_requested(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.dropna(subset=usecols)
        mask = frame[text_col].astype(str).str.strip() != ""
        if text_col_b:
            mask &= frame[text_col_b].astype(str).str.strip() != ""
        return frame.loc[mask]

    if limit is None:
        df = _keep_requested(pd.read_csv(path, usecols=usecols))
    else:
        parts = []
        kept = 0
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=max(limit, 4096)):
            chunk = _keep_requested(chunk)
            if chunk.empty:
                continue
            remain = limit - kept
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]
            parts.append(chunk)
            kept += len(chunk)
            if kept >= limit:
                break
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)

    texts_b = df[text_col_b].astype(str).tolist() if text_col_b else None
    return df[text_col].astype(str).tolist(), df[label_col].tolist(), texts_b


def _load_hf(
    dataset_id: str,
    split: str,
    text_col: str,
    label_col: str,
    limit: Optional[int],
    config: Optional[str],
    text_col_b: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[List[object], List[object], Optional[List[str]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError(
            "--hf requires the optional text extra: "
            "pip install 'dataset-complexity-profiler[text]'"
        ) from exc

    if limit is not None and limit < 1:
        raise ValueError(f"Limit must be >= 1, got {limit}")
    kwargs = {"split": split}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        if config:
            ds = load_dataset(dataset_id, config, **kwargs)
        else:
            ds = load_dataset(dataset_id, **kwargs)
    except Exception as exc:
        if "trust_remote_code" in str(exc) and not trust_remote_code:
            raise ValueError(
                f"HuggingFace dataset '{dataset_id}' requires remote code execution. "
                "Pass --trust-remote-code to allow this."
            ) from exc
        raise

    def _keep_row(row: dict) -> bool:
        if not _nonempty_text(row[text_col]) or _is_missing(row[label_col]):
            return False
        if text_col_b and not _nonempty_text(row[text_col_b]):
            return False
        return True

    ds = ds.filter(_keep_row)
    if limit is not None and len(ds) > limit:
        ds = ds.shuffle(seed=42).select(range(limit))
    texts_b = list(ds[text_col_b]) if text_col_b else None
    return list(ds[text_col]), list(ds[label_col]), texts_b


def cmd_analyze(args: argparse.Namespace) -> int:
    from dataset_complexity_profiler import DatasetProfiler
    from dataset_complexity_profiler.defaults import DIM_BUCKETS

    try:
        profiler = DatasetProfiler(auto_load_meta_model=not args.no_meta_model)

        texts_b = None
        if args.csv:
            texts, labels, texts_b = _read_csv_texts_labels(
                Path(args.csv),
                args.text_col,
                args.label_col,
                args.limit,
                text_col_b=args.text_col_b,
            )
            dataset_name = args.name or Path(args.csv).stem
        elif args.hf:
            texts, labels, texts_b = _load_hf(
                args.hf,
                args.split,
                args.text_col,
                args.label_col,
                args.limit,
                args.config,
                text_col_b=args.text_col_b,
                trust_remote_code=args.trust_remote_code,
            )
            dataset_name = args.name or args.hf
        else:
            raise ValueError("Provide --csv PATH or --hf DATASET_ID")

        embedder_name = args.embedder or DEFAULT_EMBEDDER_NAME
        report = profiler.analyze_text_dataset(
            texts,
            labels,
            dataset_name=dataset_name,
            embedder_name=embedder_name,
            batch_size=args.batch_size,
            sample_limit=None,
            quality_threshold=args.quality_threshold,
            task_family="single" if args.task == "classification" else args.task,
            texts_b=texts_b,
            empirical=args.empirical,
        )
    except _CLI_USER_ERRORS as exc:
        print(f"Error: {exc}")
        return 1

    dim = report["recommended_embedding_dim"]
    method = report.get("method") or "predictive"
    shape = (report["sample_count"], report["original_dim"])
    print(
        f"Dataset profiling complete. Shape: {shape}. "
        f"Recommended PCA dimension: {dim} (Method: {method})."
    )
    if int(dim) not in DIM_BUCKETS or report.get("dim_is_bucket") is False:
        print("Note: Output is an exact dimension bound, not a standard bucket.")
    metrics = {
        "dataset": report["dataset_name"],
        "baseline_quality": report["baseline_quality"],
        "class_count": report["class_count"],
        **(report.get("complexity_profile") or {}),
        "recommended_embedding_dim": dim,
        "method": method,
    }
    print(metrics)

    if args.output:
        out = Path(args.output)
        try:
            with out.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        except OSError as exc:
            print(f"Error: could not write report to {out}: {exc}", file=sys.stderr)
            return 1
        print(f"Full report saved to {out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcp",
        description="Dataset Complexity Profiler — profile text classification datasets",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Profile a dataset and recommend adaptation")
    src = analyze.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to CSV with text and label columns")
    src.add_argument("--hf", help="HuggingFace dataset id")
    analyze.add_argument("--config", default=None, help="HF dataset config name")
    analyze.add_argument("--split", default="train")
    analyze.add_argument("--text-col", default="text")
    analyze.add_argument("--text-col-b", default=None, help="Second text column for pair/STS")
    analyze.add_argument("--label-col", default="label")
    analyze.add_argument(
        "--task",
        default="classification",
        choices=["classification", "pair", "sts"],
        help="Probe family: classification (single), pair, sts",
    )
    analyze.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows/examples (default: full file / full HF split)",
    )
    analyze.add_argument("--name", default=None)
    analyze.add_argument("--embedder", default=None, help="SentenceTransformer name (default: multilingual MiniLM)")
    analyze.add_argument("--batch-size", type=int, default=64)
    analyze.add_argument("--quality-threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD)
    analyze.add_argument("--output", default=None, help="Save full JSON report")
    analyze.add_argument(
        "--no-meta-model",
        action="store_true",
        help="Skip loading packaged meta-model (empirical only)",
    )
    analyze.add_argument(
        "--empirical",
        action="store_true",
        help="Force empirical PCA grid search (slow; ignores the packaged meta-model)",
    )
    analyze.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow HuggingFace to execute custom dataset scripts",
    )
    analyze.set_defaults(func=cmd_analyze)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except _CLI_USER_ERRORS as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

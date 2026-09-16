"""Train the packaged Random Forest classifier on dimension buckets and save artifact v2."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_predict

from dataset_complexity_profiler.defaults import (
    COLLECT_MIN_LINEAR_KAPPA,
    COLLECT_MIN_STS_ABS_SPEARMAN,
    DIM_BUCKET_BINS,
    DIM_BUCKETS,
)
from dataset_complexity_profiler.meta_artifact import (
    ID_COLUMN,
    INTERPRETABLE_FEATURE_NAMES,
    LINEAR_SEP_COLUMNS,
    LINEAR_SEP_METRIC_COLUMN,
    LINEAR_SEP_THRESHOLD_COLUMN,
    LINEAR_SEP_VALUE_COLUMN,
    TARGET_COLUMN,
    save_artifact,
    wrap_artifact,
)

TARGET_CLASSES: List[int] = list(DIM_BUCKETS)
TARGET_BINS: List[int] = list(DIM_BUCKET_BINS)


def _base_source_name(name: object) -> str:
    if pd.isna(name):
        return ""
    return str(name).split("[", 1)[0].strip().lower()


def _resolve_separability_thresholds(
    df: pd.DataFrame,
    min_kappa: float,
    min_sts_abs_spearman: float,
) -> pd.Series:
    """Per-row gate: stored threshold from collect, else the default for that metric."""
    if LINEAR_SEP_METRIC_COLUMN in df.columns:
        metric = df[LINEAR_SEP_METRIC_COLUMN].fillna("").astype(str)
    else:
        metric = pd.Series("", index=df.index, dtype=object)
    fallback = pd.Series(
        np.where(
            metric.str.contains("spearman", case=False, na=False),
            float(min_sts_abs_spearman),
            float(min_kappa),
        ),
        index=df.index,
        dtype=np.float64,
    )
    if LINEAR_SEP_THRESHOLD_COLUMN not in df.columns:
        return fallback
    stored = pd.to_numeric(df[LINEAR_SEP_THRESHOLD_COLUMN], errors="coerce")
    stored = stored.where(stored > 0.0)
    return stored.fillna(fallback)


def _drop_failed_separability_rows(
    df: pd.DataFrame,
    min_kappa: float = COLLECT_MIN_LINEAR_KAPPA,
    min_sts_abs_spearman: float = COLLECT_MIN_STS_ABS_SPEARMAN,
) -> pd.DataFrame:
    """Numeric gate: drop the row when ``value < threshold``."""
    if LINEAR_SEP_VALUE_COLUMN in df.columns:
        values = pd.to_numeric(df[LINEAR_SEP_VALUE_COLUMN], errors="coerce")
    else:
        values = pd.Series(np.nan, index=df.index, dtype=np.float64)
    thresholds = _resolve_separability_thresholds(df, min_kappa, min_sts_abs_spearman)

    failed = values.notna() & (values < thresholds)
    if failed.any():
        dropped = (
            sorted(set(df.loc[failed, ID_COLUMN].map(_base_source_name)))
            if ID_COLUMN in df.columns
            else []
        )
        print(
            f"Dropped {int(failed.sum())} rows below the linear-separability gate"
            + (f": {', '.join(dropped)}" if dropped else "")
        )
        return df.loc[~failed].copy()
    return df


def _require_numeric_separability_gate(df: pd.DataFrame) -> None:
    if LINEAR_SEP_VALUE_COLUMN not in df.columns:
        raise ValueError(
            f"CSV is missing numeric linear-separability columns "
            f"({LINEAR_SEP_VALUE_COLUMN}); collect or recompute the gate first."
        )


def _assign_target_buckets(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    out = df.copy()
    bins = list(TARGET_BINS)
    labels = list(TARGET_CLASSES)
    n_intervals = len(bins) - 1
    if n_intervals > len(labels):
        labels = labels + [labels[-1]] * (n_intervals - len(labels))

    indices = pd.cut(
        pd.to_numeric(out[target_column], errors="coerce"),
        bins=bins,
        labels=False,
    )
    before = out.shape[0]
    mask = indices.notna()
    out = out.loc[mask].copy()
    valid_indices = indices.loc[mask].astype(int)
    out["target_class"] = [labels[int(i)] for i in valid_indices.to_numpy()]

    dropped = before - out.shape[0]
    if dropped:
        print(f"Dropped {dropped} rows outside dim buckets {list(TARGET_CLASSES)}")
    counts = out["target_class"].value_counts().sort_index()
    print("Bucket counts: " + ", ".join(f"{int(k)}={int(v)}" for k, v in counts.items()))
    return out


def _adjacent_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[int] = TARGET_CLASSES,
) -> float:
    """Fraction of predictions no farther than ±1 bucket."""
    index = {int(c): i for i, c in enumerate(classes)}
    hits = 0
    n = 0
    for true, pred in zip(y_true, y_pred):
        ti = index.get(int(true))
        pi = index.get(int(pred))
        n += 1
        if ti is not None and pi is not None and abs(ti - pi) <= 1:
            hits += 1
    return float(hits / n) if n else 0.0


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


def _cv_predict(model, X: np.ndarray, y: np.ndarray, cv_splitter: object, groups) -> np.ndarray:
    kwargs: dict = {"cv": cv_splitter}
    if groups is not None:
        kwargs["groups"] = groups
    return cross_val_predict(model, X, y, **kwargs)


def _fallback_cv(n_splits: int, y: np.ndarray) -> object:
    n_splits = max(2, int(n_splits))
    _, counts = np.unique(y, return_counts=True)
    if len(counts) >= 2 and np.min(counts) >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def _iter_splits(cv_splitter: object, X: np.ndarray, y: np.ndarray, groups):
    if groups is not None:
        return list(cv_splitter.split(X, y, groups))
    return list(cv_splitter.split(X, y))


def _oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    cv_splitter: object,
    groups,
) -> Tuple[np.ndarray, dict]:
    """GroupKFold OOF for the packaged RF."""
    model = _make_rf()
    oof = np.asarray(_cv_predict(model, X, y, cv_splitter, groups))
    fold_acc: List[float] = []
    fold_adj: List[float] = []
    for _, test_idx in _iter_splits(cv_splitter, X, y, groups):
        fold_acc.append(float(accuracy_score(y[test_idx], oof[test_idx])))
        fold_adj.append(_adjacent_accuracy(y[test_idx], oof[test_idx]))
    stats = {
        "accuracy": float(accuracy_score(y, oof)),
        "adjacent_accuracy": float(_adjacent_accuracy(y, oof)),
        "fold_accuracy": fold_acc,
        "fold_adjacent_accuracy": fold_adj,
        "cv_accuracy_mean": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_adjacent_mean": float(np.mean(fold_adj)),
        "cv_adjacent_std": float(np.std(fold_adj)),
    }
    return oof, stats


def train_meta_model(
    selection_csv: str = "research/feature_selection.csv",
    output_model: str = "src/dataset_complexity_profiler/meta_model.skops",
    cv_folds: int = 5,
    train_extra_csvs: Optional[List[str]] = None,
    task_families: Optional[List[str]] = None,
) -> Optional[dict]:
    """Fit an RF on dim buckets. Extra CSV is concatenated first and joins CV with selection."""
    csv_path = Path(selection_csv)
    if not csv_path.is_file():
        print(f"Error: {selection_csv} not found.")
        return None

    needed_cols = {
        ID_COLUMN,
        TARGET_COLUMN,
        "recommended_embedding_dim",
        "task_family",
        *INTERPRETABLE_FEATURE_NAMES,
        *LINEAR_SEP_COLUMNS,
    }
    df = pd.read_csv(csv_path, usecols=lambda c: c in needed_cols)
    for extra in train_extra_csvs or []:
        extra_path = Path(extra)
        if extra_path.is_file():
            extra_df = pd.read_csv(extra_path, usecols=lambda c: c in needed_cols)
            print(f"Concat training extra from {extra_path} ({extra_df.shape[0]} rows)")
            df = pd.concat([df, extra_df], ignore_index=True, sort=False)
        else:
            raise FileNotFoundError(
                f"training extra CSV not found: {extra}. "
                "The packaged model is trained on selection+extra; "
                "omitting it silently would produce a different artifact."
            )

    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first") if ID_COLUMN in df.columns else df
    _require_numeric_separability_gate(df)
    df = _drop_failed_separability_rows(df)

    families = task_families or ["single", "pair", "sts"]
    if "task_family" in df.columns:
        before = df.shape[0]
        family_str = df["task_family"].fillna("single").astype(str)
        family_str = family_str.replace(
            {
                "0": "single",
                "0.0": "single",
                "1": "pair",
                "1.0": "pair",
                "4": "tabular",
                "4.0": "tabular",
                "nan": "single",
            }
        )
        df = df[family_str.isin(families)]
        print(f"Filtered task_families={families}: {before} → {df.shape[0]} rows")

    if TARGET_COLUMN in df.columns:
        target_column = TARGET_COLUMN
    elif "recommended_embedding_dim" in df.columns:
        target_column = "recommended_embedding_dim"
    else:
        raise ValueError(
            "Expected 'recommended_dim' or 'recommended_embedding_dim' column"
        )

    before_target = df.shape[0]
    df = df.dropna(subset=[target_column])
    dropped_target = before_target - df.shape[0]
    if dropped_target:
        print(f"Dropped {dropped_target} rows with missing '{target_column}'")

    df = _assign_target_buckets(df, target_column)

    missing = [name for name in INTERPRETABLE_FEATURE_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"Missing interpretable features: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df[list(INTERPRETABLE_FEATURE_NAMES)] = df[list(INTERPRETABLE_FEATURE_NAMES)].fillna(0.0)

    if df.shape[0] < 2:
        print("Error: Need at least 2 rows to train the model.")
        return None

    y = df["target_class"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        print("Error: Target contains only one class after filtering. Cannot train RF.")
        return None

    majority = int(pd.Series(y).mode().iloc[0])
    majority_acc = float(np.mean(y == majority))

    feature_columns = list(INTERPRETABLE_FEATURE_NAMES)
    X = np.nan_to_num(
        df[feature_columns].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    print(f"Loaded {df.shape[0]} rows × {len(feature_columns)} features: {', '.join(feature_columns)}")
    print(f"Majority-class baseline (always {majority}): accuracy={majority_acc:.4f}")

    groups = None
    n_splits = min(cv_folds, max(df.shape[0] - 1, 2))
    if ID_COLUMN in df.columns:
        groups = df[ID_COLUMN].map(_base_source_name).to_numpy()
        n_groups = len(set(groups.tolist()))
        n_splits = min(n_splits, n_groups)
        if n_splits >= 2:
            cv_splitter: object = GroupKFold(n_splits=n_splits)
            print(f"Using GroupKFold(n_splits={n_splits}) by source ({n_groups} groups)")
        else:
            groups = None
            cv_splitter = _fallback_cv(n_splits, y)
    else:
        cv_splitter = _fallback_cv(n_splits, y)

    _, oof_stats = _oof_predictions(X, y, cv_splitter, groups)
    print(
        f"OOF RF: accuracy={oof_stats['accuracy']:.4f}  "
        f"adjacent_accuracy={oof_stats['adjacent_accuracy']:.4f}"
    )
    print(f"Per-fold accuracy: {oof_stats['fold_accuracy']}")
    print(f"Per-fold adjacent accuracy: {oof_stats['fold_adjacent_accuracy']}")

    meta_model = _make_rf()
    meta_model.fit(X, y)

    artifact = wrap_artifact(
        meta_model,
        feature_names=feature_columns,
    )
    output_path = Path(output_model)
    save_artifact(artifact, output_path)
    print(f"Saved artifact v2 to {output_path}")

    return {
        "n_examples": X.shape[0],
        "n_features": X.shape[1],
        "feature_names": list(feature_columns),
        "target_classes": list(TARGET_CLASSES),
        "accuracy": float(oof_stats["accuracy"]),
        "adjacent_accuracy": float(oof_stats["adjacent_accuracy"]),
        "cv_accuracy_mean": float(oof_stats["cv_accuracy_mean"]),
        "cv_accuracy_std": float(oof_stats["cv_accuracy_std"]),
        "cv_adjacent_mean": float(oof_stats["cv_adjacent_mean"]),
        "majority_accuracy": majority_acc,
        "output_model": str(output_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier on PCA-dimension buckets"
    )
    parser.add_argument(
        "--selection-csv",
        default="research/feature_selection.csv",
        help="feature-selection table (PyMFE present); also the base of the train concat",
    )
    parser.add_argument(
        "--output-model",
        default="src/dataset_complexity_profiler/meta_model.skops",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--train-extra-csv",
        action="append",
        default=[],
        help="rows without PyMFE; concatenated with selection before CV and the packaged RF",
    )
    parser.add_argument("--task-families", default="single,pair,sts")
    args = parser.parse_args()
    families = [x.strip() for x in args.task_families.split(",") if x.strip()]
    summary = train_meta_model(
        selection_csv=args.selection_csv,
        output_model=args.output_model,
        cv_folds=args.cv_folds,
        train_extra_csvs=args.train_extra_csv,
        task_families=families,
    )
    raise SystemExit(0 if summary else 1)

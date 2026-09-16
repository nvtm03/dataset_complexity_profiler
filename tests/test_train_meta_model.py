from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset_complexity_profiler.meta_artifact import (
    INTERPRETABLE_FEATURE_NAMES,
    TARGET_COLUMN,
)
from dataset_complexity_profiler.probes import adjusted_accuracy_gain


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "research" / "train_meta_model.py"
    spec = importlib.util.spec_from_file_location("train_meta_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _minimal_row(**overrides):
    row = {
        "dataset_name": "Toy [baseline]",
        TARGET_COLUMN: 16.0,
        "class_count": 2.0,
        "baseline_quality": 0.9,
        "mean_centroid_cosine_distance": 0.2,
        "intrinsic_dim_twonn": 4.0,
        "intrinsic_dim_pca_95": 8.0,
        "minority_fraction": 0.5,
        "task_family": "single",
        "linear_sep_metric": "kappa_vs_majority",
        "linear_sep_value": 0.8,
        "linear_sep_threshold": 0.10,
    }
    row.update(overrides)
    return row


def test_training_extra_csv_is_present():
    from pathlib import Path

    import pandas as pd

    extra = Path(__file__).resolve().parents[1] / "research" / "training_extra.csv"
    notebook = Path(__file__).resolve().parents[1] / "research" / "feature_selection.ipynb"
    collect = Path(__file__).resolve().parents[1] / "research" / "collect_training_extra.py"
    assert extra.is_file(), "research/training_extra.csv must be in the repo"
    assert notebook.is_file()
    assert collect.is_file()
    df = pd.read_csv(extra)
    assert df.shape[0] >= 50
    for name in INTERPRETABLE_FEATURE_NAMES:
        assert name in df.columns


def test_missing_train_extra_csv_is_an_error(tmp_path):
    tmm = _load_train_module()
    csv_path = tmp_path / "ok.csv"
    pd.DataFrame([_minimal_row(dataset_name=f"Src{i}") for i in range(8)]).to_csv(
        csv_path, index=False
    )
    missing = tmp_path / "no-such-extra.csv"
    with pytest.raises(FileNotFoundError, match="training extra CSV not found"):
        tmm.train_meta_model(
            selection_csv=str(csv_path),
            output_model=str(tmp_path / "out.skops"),
            cv_folds=2,
            train_extra_csvs=[str(missing)],
        )


def test_train_fail_fast_without_linear_sep_column(tmp_path):
    tmm = _load_train_module()
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "dataset_name": "A [x]",
                TARGET_COLUMN: 16.0,
                **{name: 1.0 for name in INTERPRETABLE_FEATURE_NAMES},
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="linear-separability"):
        tmm.train_meta_model(
            selection_csv=str(csv_path),
            output_model=str(tmp_path / "out.skops"),
            cv_folds=2,
        )


def test_drop_failed_separability_uses_numeric_gate():
    tmm = _load_train_module()
    df = pd.DataFrame(
        [
            _minimal_row(dataset_name="Easy [a]", linear_sep_value=0.8),
            _minimal_row(
                dataset_name="Hard [a]",
                linear_sep_value=0.01,
                baseline_quality=0.05,
            ),
        ]
    )
    out = tmm._drop_failed_separability_rows(df)
    assert list(out["dataset_name"]) == ["Easy [a]"]


def test_missing_target_dropped_before_fillna(tmp_path, monkeypatch):
    tmm = _load_train_module()
    rows = []
    for i in range(12):
        rows.append(
            _minimal_row(
                dataset_name=f"Src{i} [v]",
                **{name: float(i + 1) for name in INTERPRETABLE_FEATURE_NAMES},
                **{
                    TARGET_COLUMN: float("nan") if i == 0 else 8.0 + i,
                    "linear_sep_value": 0.5,
                },
            )
        )
    csv_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    captured = {}

    def fake_nested(X, y, *args, **kwargs):
        captured["n"] = len(y)
        pred = np.full(len(y), int(y[0]))
        stats = {
            "accuracy": 1.0,
            "adjacent_accuracy": 1.0,
            "fold_accuracy": [1.0, 1.0],
            "fold_adjacent_accuracy": [1.0, 1.0],
            "cv_accuracy_mean": 1.0,
            "cv_accuracy_std": 0.0,
            "cv_adjacent_mean": 1.0,
            "cv_adjacent_std": 0.0,
        }
        return pred, stats

    monkeypatch.setattr(tmm, "_oof_predictions", fake_nested)

    summary = tmm.train_meta_model(
        selection_csv=str(csv_path),
        output_model=str(tmp_path / "out.skops"),
        cv_folds=2,
    )
    assert summary is not None
    assert captured["n"] == 11
    assert summary["n_examples"] == 11
    assert "accuracy" in summary
    assert "adjacent_accuracy" in summary


def test_adjacent_accuracy_allows_one_bucket_error():
    tmm = _load_train_module()
    y = np.array([8, 16, 32, 64])
    pred = np.array([8, 32, 16, 256])
    assert tmm._adjacent_accuracy(y, pred) == 0.75


def test_assign_target_buckets_cuts_empirical_grid_edges():
    from dataset_complexity_profiler.defaults import DIM_BUCKET_BINS, DIM_BUCKETS

    tmm = _load_train_module()
    assert DIM_BUCKET_BINS == (0, 4, 8, 16, 32, 64, 128, 256, 100000)
    assert DIM_BUCKETS == (4, 8, 16, 32, 64, 128, 256)
    df = pd.DataFrame({TARGET_COLUMN: [12, 24, 48, 5, 6, 192, 193, 1001]})
    out = tmm._assign_target_buckets(df, TARGET_COLUMN)
    assert list(out["target_class"]) == [16, 32, 64, 8, 8, 256, 256, 256]


def test_train_csv_load_skips_unused_pymfe_columns(tmp_path, monkeypatch):
    tmm = _load_train_module()
    rows = [_minimal_row(dataset_name=f"Src{i} [v]", **{TARGET_COLUMN: 8.0 + i}) for i in range(8)]
    csv_path = tmp_path / "wide.csv"
    pd.DataFrame(rows).assign(**{"f1.mean": 0.1, "n1": 0.2}).to_csv(csv_path, index=False)

    seen = []
    orig = tmm.pd.read_csv

    def wrapped(*args, **kwargs):
        seen.append(kwargs.get("usecols"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(tmm.pd, "read_csv", wrapped)
    monkeypatch.setattr(
        tmm,
        "_oof_predictions",
        lambda X, y, *a, **k: (
            np.full(len(y), int(y[0])),
            {
                "accuracy": 1.0,
                "adjacent_accuracy": 1.0,
                "fold_accuracy": [1.0],
                "fold_adjacent_accuracy": [1.0],
                "cv_accuracy_mean": 1.0,
                "cv_accuracy_std": 0.0,
                "cv_adjacent_mean": 1.0,
                "cv_adjacent_std": 0.0,
            },
        ),
    )
    summary = tmm.train_meta_model(
        selection_csv=str(csv_path),
        output_model=str(tmp_path / "out.skops"),
        cv_folds=2,
    )
    assert summary is not None
    assert seen and callable(seen[0])
    assert seen[0]("class_count") is True
    assert seen[0]("f1.mean") is False
    assert seen[0]("n1") is False


def test_fallback_cv_uses_kfold_when_a_class_is_too_rare():
    tmm = _load_train_module()
    y = np.array([0] * 10 + [1] * 1)
    cv = tmm._fallback_cv(3, y)
    assert type(cv).__name__ == "KFold"
    y_ok = np.array([0] * 10 + [1] * 10)
    cv_ok = tmm._fallback_cv(2, y_ok)
    assert type(cv_ok).__name__ == "StratifiedKFold"


def test_reconstructed_binary_kappa_matches_probe():
    baseline = 0.85
    minority = 0.3
    majority = 1.0 - minority
    assert abs(adjusted_accuracy_gain(baseline, majority) - (0.15 / 0.3)) < 1e-9

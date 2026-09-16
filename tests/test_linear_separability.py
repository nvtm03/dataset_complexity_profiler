import numpy as np

from dataset_complexity_profiler.meta_artifact import compact_feature_columns
from dataset_complexity_profiler.probes import (
    adjusted_accuracy_gain,
    linear_separability_status,
    majority_chance,
)


def test_majority_chance_and_adjusted_accuracy_gain():
    y = [0] * 70 + [1] * 30
    assert abs(majority_chance(y) - 0.7) < 1e-9
    assert adjusted_accuracy_gain(1.0, 1.0) == 1.0
    assert adjusted_accuracy_gain(0.5, 1.0) == 0.0
    gain = adjusted_accuracy_gain(0.85, 0.7)
    assert abs(gain - (0.15 / 0.3)) < 1e-9


def test_goemotions_like_fails_kappa_gate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 200, size=400)
    status = linear_separability_status(0.02, y, task_family="single", min_kappa=0.10)
    assert status["passed"] is False
    assert "failed linear separability threshold" in status["message"]
    assert "adjusted accuracy gain vs majority" in status["message"]
    assert "Cohen" not in status["message"]
    assert status["metric"] == "adjusted_accuracy_gain"


def test_require_linear_separability_raises_on_failed_gate():
    import pytest

    from dataset_complexity_profiler.probes import require_linear_separability

    status = linear_separability_status(0.02, [0] * 70 + [1] * 30, task_family="single")
    assert status["passed"] is False
    with pytest.raises(ValueError, match="Cannot recommend a PCA dimension"):
        require_linear_separability(status, dataset_name="QQP")


def test_easy_binary_passes_kappa_gate():
    y = [0] * 50 + [1] * 50
    status = linear_separability_status(0.92, y, task_family="single")
    assert status["passed"] is True
    assert status["message"] is None


def test_sts_gate_uses_abs_spearman():
    y = np.linspace(0, 5, 40)
    fail = linear_separability_status(0.05, y, task_family="sts")
    assert fail["passed"] is False
    ok = linear_separability_status(-0.4, y, task_family="sts")
    assert ok["passed"] is True
    assert ok["metric"] == "abs_spearman"


def test_compact_feature_columns_are_interpretable():
    cols = [
        "class_count",
        "baseline_quality",
        "mean_centroid_cosine_distance",
        "intrinsic_dim_twonn",
        "intrinsic_dim_pca_95",
        "f1.mean",
        "n1",
        "iq_range.mean",
    ]
    kept = compact_feature_columns(cols)
    assert kept == [
        "class_count",
        "baseline_quality",
        "mean_centroid_cosine_distance",
        "intrinsic_dim_twonn",
        "intrinsic_dim_pca_95",
    ]


def test_shipped_csv_has_numeric_separability_columns():
    from pathlib import Path

    import pandas as pd

    from dataset_complexity_profiler.meta_artifact import LINEAR_SEP_COLUMNS

    csv_path = Path(__file__).resolve().parents[1] / "research" / "feature_selection.csv"
    df = pd.read_csv(csv_path, nrows=5)
    for col in LINEAR_SEP_COLUMNS:
        assert col in df.columns
    from dataset_complexity_profiler.meta_artifact import INTERPRETABLE_FEATURE_NAMES

    for name in INTERPRETABLE_FEATURE_NAMES:
        assert name in df.columns

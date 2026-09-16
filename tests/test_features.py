import numpy as np
import pytest

from dataset_complexity_profiler import INTERPRETABLE_FEATURE_NAMES, DatasetProfiler
from dataset_complexity_profiler.features import (
    build_meta_feature_vector,
    evaluate_quality,
    named_feature_vector,
    prepare_data,
)
from dataset_complexity_profiler.probes import sts_scorer


def test_named_feature_vector_rejects_length_mismatch():
    try:
        named_feature_vector(np.array([1.0, 2.0]), ["a", "b", "c"])
    except ValueError as exc:
        assert "length" in str(exc).lower()
        return
    raise AssertionError("expected ValueError for name/value length mismatch")


def test_prepare_data_keeps_sts_float_scores():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 8))
    y = np.array([0.1, 0.4, 1.2, 2.7] + list(rng.uniform(0, 5, size=16)))
    _, y_arr = prepare_data(X, y, task_family="sts")
    assert y_arr.dtype.kind == "f"
    assert abs(float(y_arr[0]) - 0.1) < 1e-12
    assert abs(float(y_arr[1]) - 0.4) < 1e-12
    assert abs(float(y_arr[2]) - 1.2) < 1e-12
    assert abs(float(y_arr[3]) - 2.7) < 1e-12


def test_prepare_data_rejects_empty_and_single_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(0, 8))
    with pytest.raises(ValueError, match="empty"):
        prepare_data(X, np.array([]))

    X2 = rng.normal(size=(12, 8))
    with pytest.raises(ValueError, match="at least 2 classes"):
        prepare_data(X2, np.zeros(12))

    with pytest.raises(ValueError, match="length mismatch"):
        prepare_data(rng.normal(size=(10, 5)), np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="Unknown task_family"):
        prepare_data(rng.normal(size=(20, 5)), np.array([0] * 10 + [1] * 10), task_family="banana")


def test_prepare_data_maps_inf_to_finite_and_rejects_odd_sts_width():
    from dataset_complexity_profiler.defaults import ODD_PAIR_STS_FEATURES_MSG

    X = np.array([[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    X_clean, _ = prepare_data(X, y)
    assert np.isfinite(X_clean).all()

    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="even number of features"):
        prepare_data(rng.normal(size=(20, 5)), rng.uniform(0, 1, size=20), task_family="sts")
    with pytest.raises(ValueError, match="even number of features"):
        prepare_data(rng.normal(size=(20, 7)), np.array([0] * 10 + [1] * 10), task_family="pair")
    assert "text A and text B" in ODD_PAIR_STS_FEATURES_MSG


def test_prepare_data_drops_missing_labels_without_phantom_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.array([0] * 15 + [1] * 15 + [2] * 10, dtype=np.float64)
    y[5:12] = np.nan
    with pytest.warns(UserWarning, match="missing labels"):
        _, y_arr = prepare_data(X, y)
    assert len(y_arr) == 33
    assert len(np.unique(y_arr)) == 3
    assert int(np.min(y_arr)) >= 0


def test_prepare_data_drops_nan_mixed_with_string_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = ["a"] * 15 + [np.nan] * 10 + ["b"] * 15
    with pytest.warns(UserWarning, match="missing labels"):
        _, y_arr = prepare_data(X, y)
    assert len(y_arr) == 30
    assert np.unique(y_arr).size == 2


def test_prepare_data_accepts_dataframe_and_series_labels():
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y_df = pd.DataFrame({"label": ["a"] * 20 + ["b"] * 20})
    _, y_from_df = prepare_data(X, y_df)
    _, y_from_series = prepare_data(X, y_df["label"])
    assert y_from_df.shape[0] == 40
    assert y_from_series.shape[0] == 40
    assert np.unique(y_from_df).size == 2
    assert np.array_equal(y_from_df, y_from_series)


def test_prepare_data_drops_stringified_nan_from_numpy_str_array():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.array(["a"] * 15 + [np.nan] * 10 + ["b"] * 15)
    assert y.dtype.kind in "US"
    assert "nan" in set(y.tolist())
    with pytest.warns(UserWarning, match="missing labels"):
        _, y_arr = prepare_data(X, y)
    assert len(y_arr) == 30
    assert np.unique(y_arr).size == 2


def test_prepare_data_keeps_none_as_a_valid_nlp_label():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.array(["sports", "none"] * 20)
    _, y_arr = prepare_data(X, y)
    assert y_arr.shape[0] == 40
    assert np.unique(y_arr).size == 2


def test_evaluate_quality_does_not_fallback_sts_to_ridge():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.uniform(0, 1, size=20)
    assert evaluate_quality(X, y, scorer=sts_scorer) == 0.0


def test_prepare_data_remaps_negative_class_ids():
    profiler = DatasetProfiler(auto_load_meta_model=False)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 16))
    y = np.array([-1] * 10 + [0] * 10 + [1] * 10)
    _, y_arr = prepare_data(X, y)
    assert int(np.min(y_arr)) >= 0
    assert len(np.unique(y_arr)) == 3
    feats = profiler.build_meta_feature_vector(X, y)
    assert list(feats) == list(INTERPRETABLE_FEATURE_NAMES)
    assert all(np.isfinite(value) for value in feats.values())


def test_gapped_integer_labels_keep_two_classes():
    """Labels {0, 7} must not look like an empty minority class (old np.bincount bug)."""
    profiler = DatasetProfiler(auto_load_meta_model=False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    y = np.array([0] * 20 + [7] * 20)
    feats = profiler.build_meta_feature_vector(X, y)
    assert feats["class_count"] == 2.0


def test_prepare_data_casts_float16_and_float32_to_float64():
    rng = np.random.default_rng(0)
    y = np.array([0] * 10 + [1] * 10)
    for dtype in (np.float16, np.float32):
        X = rng.normal(size=(20, 6)).astype(dtype)
        X_clean, _ = prepare_data(X, y)
        assert X_clean.dtype == np.float64


def test_centroid_distance_computed_above_1000_classes(monkeypatch):
    from dataset_complexity_profiler import features as feat_mod

    monkeypatch.setattr(feat_mod, "evaluate_quality", lambda *_a, **_k: 0.5)
    rng = np.random.default_rng(0)
    n_classes = 1001
    n = 1002
    y = np.arange(n) % n_classes
    X = rng.normal(size=(n, 8))
    feats = build_meta_feature_vector(X, y)
    assert feats["class_count"] == float(n_classes)
    assert feats["mean_centroid_cosine_distance"] > 0.0

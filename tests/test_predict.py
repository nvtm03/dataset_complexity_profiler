import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from dataset_complexity_profiler import INTERPRETABLE_FEATURE_NAMES, DatasetProfiler
from dataset_complexity_profiler.defaults import DIM_BUCKETS, feasible_dim_bucket
from dataset_complexity_profiler.meta_artifact import save_artifact, wrap_artifact
from dataset_complexity_profiler.predict import conservative_dim_from_proba, is_recoverable_predict_error


def _two_blobs(n=40, d=16, seed=0, sep=4.0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X = np.vstack(
        [
            rng.normal(0.0, 0.4, size=(n0, d)),
            rng.normal(sep, 0.4, size=(n - n0, d)),
        ]
    )
    y = np.array([0] * n0 + [1] * (n - n0))
    return X, y


def _sts_related(n=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, dim))
    scores = np.linspace(0.0, 1.0, n)
    v = u * scores[:, None] + rng.normal(scale=0.05, size=u.shape) * (1.0 - scores[:, None])
    return np.hstack([u, v]), scores


def test_feasible_dim_bucket_returns_cap_below_bucket_rank():
    assert feasible_dim_bucket(256, n_features=1, n_samples=40) == 1
    assert feasible_dim_bucket(256, n_features=8, n_samples=3) == 2


def test_predict_rejects_too_few_samples():
    from dataset_complexity_profiler.defaults import MIN_SAMPLES_FOR_DIM_RECOMMENDATION

    profiler = DatasetProfiler(auto_load_meta_model=True)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(4, 384))
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match=str(MIN_SAMPLES_FOR_DIM_RECOMMENDATION)):
        profiler.predict_embedding_dim(X, y)
    with pytest.raises(ValueError, match=str(MIN_SAMPLES_FOR_DIM_RECOMMENDATION)):
        profiler.analyze_and_adapt(X, y)


def test_predict_rejects_foreign_feature_names():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 16))
    y = np.array([0] * 20 + [1] * 20)
    with pytest.raises(ValueError, match="mismatch"):
        profiler.predict_embedding_dim(
            X, y, features={"foo": 1.0, "bar": 2.0}, feature_names=["foo", "bar"]
        )


def test_predict_rejects_all_zero_feature_vector():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 16))
    y = np.array([0] * 20 + [1] * 20)
    zeros = {name: 0.0 for name in INTERPRETABLE_FEATURE_NAMES}
    with pytest.raises(ValueError, match="all zeros"):
        profiler.predict_embedding_dim(X, y, features=zeros)


def test_predict_warns_when_rank_below_bucket():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    X, y = _two_blobs(n=40, d=1, seed=0, sep=6.0)
    with pytest.warns(UserWarning, match="not a standard bucket"):
        dim = profiler.predict_embedding_dim(X, y)
    assert dim == 1


def test_predict_returns_bucket_int():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    X, y = _two_blobs(n=40, d=64, seed=5)
    dim = profiler.predict_embedding_dim(X, y)
    assert isinstance(dim, int)
    assert dim in {4, 8, 16, 32, 64, 128, 256}


def test_predict_clips_bucket_to_matrix_rank():
    assert feasible_dim_bucket(256, n_features=8, n_samples=40) == 8
    assert feasible_dim_bucket(256, n_features=6, n_samples=40) == 4
    assert feasible_dim_bucket(4, n_features=64, n_samples=40) == 4
    assert feasible_dim_bucket(23, n_features=64, n_samples=40) == 16

    profiler = DatasetProfiler(auto_load_meta_model=True)
    X, y = _two_blobs(n=40, d=8, seed=5)
    dim = profiler.predict_embedding_dim(X, y)
    assert dim in {4, 8}
    assert dim <= 8


def test_sts_predict_caps_to_half_concat_width():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    X, y = _sts_related(n=40, dim=6, seed=5)
    dim = profiler.predict_embedding_dim(X, y, task_family="sts")
    assert dim <= 6
    assert dim in DIM_BUCKETS or dim < 4


def test_predict_uses_artifact_feature_names(tmp_path):
    profiler = DatasetProfiler(auto_load_meta_model=False)
    X, y = _two_blobs(n=80, d=48, seed=2)

    feats = profiler.build_meta_feature_vector(X, y)
    names = list(feats)
    vec = np.array([feats[name] for name in names], dtype=np.float64)
    meta_X = np.vstack([vec, vec * 1.01, vec * 0.99, vec + 0.1])
    meta_y = np.array([16, 16, 32, 32])

    clf = RandomForestClassifier(n_estimators=8, random_state=0)
    clf.fit(meta_X, meta_y)
    path = tmp_path / "meta.skops"
    save_artifact(wrap_artifact(clf, feature_names=names), path)

    loaded = DatasetProfiler(auto_load_meta_model=False)
    loaded.load_meta_model(str(path))
    dim = loaded.predict_embedding_dim(X, y)
    assert isinstance(dim, int)
    assert dim in DIM_BUCKETS or dim < 4
    artifact = wrap_artifact(clf, feature_names=names)
    assert "sklearn_version" in artifact
    assert "use_log_transform" not in artifact


def test_is_recoverable_predict_error_includes_mismatch():
    assert is_recoverable_predict_error(ValueError("Feature name mismatch: missing ['a']"))
    assert is_recoverable_predict_error(ValueError("Feature vector is all zeros"))
    assert is_recoverable_predict_error(ValueError("Meta model is not fitted yet."))
    assert not is_recoverable_predict_error(ValueError("Need at least 30 samples"))
    assert not is_recoverable_predict_error(ValueError("shape mismatch: value array of shape (3,)"))
    assert not is_recoverable_predict_error(IndexError("index 3"))


def test_conservative_dim_from_proba_picks_upper_quantile_not_argmax():
    classes = np.array([4, 8, 16, 32, 64, 128, 256])
    probs = np.array([0.05, 0.10, 0.40, 0.10, 0.35, 0.00, 0.00])
    assert conservative_dim_from_proba(probs, classes, cdf_threshold=0.80) == 64
    confident_on_16 = np.array([0.01, 0.02, 0.90, 0.04, 0.02, 0.01, 0.00])
    assert conservative_dim_from_proba(confident_on_16, classes, cdf_threshold=0.80) == 16


def test_conservative_dim_from_proba_uses_classes_not_hardcoded_order():
    classes = np.array([256, 4, 64, 8, 16, 32, 128])
    probs = np.array([0.00, 0.05, 0.35, 0.10, 0.40, 0.10, 0.00])
    assert conservative_dim_from_proba(probs, classes, cdf_threshold=0.80) == 64


def test_conservative_dim_from_proba_clamps_searchsorted_past_last_bin():
    classes = np.array([4, 8, 16])
    probs = np.array([0.2, 0.3, 0.5])
    assert conservative_dim_from_proba(probs, classes, cdf_threshold=1.1) == 16


def test_conservative_dim_from_proba_nan_falls_back_to_largest_bucket():
    classes = np.array([4, 8])
    probs = np.array([np.nan, 1.0])
    assert conservative_dim_from_proba(probs, classes) == 8
    assert conservative_dim_from_proba(np.zeros(3), np.array([4, 8, 16])) == 16

import numpy as np

from dataset_complexity_profiler import DatasetProfiler
from dataset_complexity_profiler.id_estimators import twonn_id


def test_twonn_tiny_sample_returns_zero():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4):
        X = rng.normal(size=(n, 8))
        value = twonn_id(X)
        assert value == 0.0
        assert np.isfinite(value)

    profiler = DatasetProfiler(auto_load_meta_model=False)
    X3 = rng.normal(size=(3, 16))
    y3 = np.array([0, 1, 0])
    feats = profiler.build_meta_feature_vector(X3, y3)
    assert feats["intrinsic_dim_twonn"] == 0.0
    assert np.isfinite(feats["intrinsic_dim_pca_95"])
    assert all(np.isfinite(v) for v in feats.values())


def test_twonn_intrinsic_dim_finite():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(120, 2))
    pad = np.zeros((120, 30))
    X = np.hstack([base, pad]) + 0.01 * rng.normal(size=(120, 32))
    id_est = twonn_id(X)
    assert np.isfinite(id_est)
    assert id_est > 0.0


def test_id_estimators_do_not_swallow_memory_error(monkeypatch):
    import pytest

    from dataset_complexity_profiler import id_estimators as id_mod

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))

    def boom(*_args, **_kwargs):
        raise MemoryError("knn graph ran out of RAM")

    monkeypatch.setattr(id_mod.NearestNeighbors, "fit", boom)
    with pytest.raises(MemoryError, match="knn graph ran out of RAM"):
        id_mod.twonn_id(X)

    monkeypatch.setattr(id_mod.PCA, "fit", boom)
    with pytest.raises(MemoryError, match="knn graph ran out of RAM"):
        id_mod.pca_id(X)

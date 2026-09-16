import numpy as np
import pytest

from dataset_complexity_profiler import DatasetProfiler
from dataset_complexity_profiler.empirical import candidate_pca_dims


def test_candidate_pca_dims_default_bound_is_two():
    dims = candidate_pca_dims(64)
    assert dims[0] == 2
    assert 4 in dims
    assert 8 in dims


def test_candidate_pca_dims_includes_small_bounds():
    dims = candidate_pca_dims(64, min_dim_bound=2, step=8)
    assert 2 in dims
    assert 4 in dims
    assert 8 in dims
    assert dims[0] == 2
    dims = candidate_pca_dims(384, min_dim_bound=12, step=8)
    assert dims[0] == 12
    assert 16 in dims
    assert 24 in dims
    assert 384 in dims
    assert dims.index(12) < dims.index(16)


def test_build_quality_curve_does_not_stop_on_near_zero_first_dim(monkeypatch):
    from dataset_complexity_profiler import empirical as empirical_mod

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16))
    y = np.linspace(0.0, 1.0, 40)

    def fake_quality(_X, _y, dim, task_family="sts"):
        return 0.01 if int(dim) <= 4 else 0.75

    monkeypatch.setattr(empirical_mod, "quality_at_dim", fake_quality)
    _, recommended, best = empirical_mod.build_quality_curve(
        X,
        y,
        [2, 4, 8, 16],
        baseline_score=0.76,
        threshold=0.97,
        task_family="sts",
        stop_at_threshold=True,
    )
    assert recommended >= 8
    assert best >= 0.75


def test_build_quality_curve_raises_when_all_evals_fail(monkeypatch):
    from dataset_complexity_profiler import empirical as empirical_mod

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16))
    y = np.array([0] * 20 + [1] * 20)

    def fake_quality(_X, _y, dim, task_family="single"):
        raise ValueError("ill-conditioned SVD")

    monkeypatch.setattr(empirical_mod, "quality_at_dim", fake_quality)
    with pytest.raises(ValueError, match="All quality_at_dim evaluations failed"):
        empirical_mod.build_quality_curve(
            X,
            y,
            [2, 4, 8],
            baseline_score=0.9,
            threshold=0.97,
        )


def test_build_quality_curve_does_not_swallow_memory_error(monkeypatch):
    from dataset_complexity_profiler import empirical as empirical_mod

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16))
    y = np.array([0] * 20 + [1] * 20)

    def fake_quality(_X, _y, dim, task_family="single"):
        raise MemoryError("pca ran out of RAM")

    monkeypatch.setattr(empirical_mod, "quality_at_dim", fake_quality)
    with pytest.raises(MemoryError, match="pca ran out of RAM"):
        empirical_mod.build_quality_curve(
            X,
            y,
            [2, 4, 8],
            baseline_score=0.9,
            threshold=0.97,
        )


def test_classification_quality_at_dim_averages_all_repeats_not_last_seed(monkeypatch):
    from dataset_complexity_profiler import empirical as empirical_mod

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = np.array([0] * 30 + [1] * 30)
    seen_states = []
    orig_splits = empirical_mod.cv_splits

    def tracking_splits(X_in, y_in, n_splits=3, random_state=42):
        seen_states.append(int(random_state))
        return orig_splits(X_in, y_in, n_splits=n_splits, random_state=random_state)

    scores = iter(0.1 * np.arange(1, 10))

    def fake_score(self, X_te, y_te):
        return float(next(scores))

    monkeypatch.setattr(empirical_mod, "cv_splits", tracking_splits)
    monkeypatch.setattr(empirical_mod.RidgeClassifier, "score", fake_score)
    result = empirical_mod.classification_quality_at_dim(X, y, dim=4, n_repeats=3)
    assert seen_states == [42, 43, 44]
    assert abs(result - 0.5) < 1e-12


def test_estimate_intrinsic_dim_rejects_tiny_separable_matrix():
    from dataset_complexity_profiler.defaults import MIN_SAMPLES_FOR_DIM_RECOMMENDATION

    profiler = DatasetProfiler(auto_load_meta_model=False)
    X = np.array([[0.0, 0.0], [8.0, 8.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match=str(MIN_SAMPLES_FOR_DIM_RECOMMENDATION)):
        profiler.estimate_intrinsic_dim(X, y)

import numpy as np

from dataset_complexity_profiler.id_estimators import estimate_id_profile
from dataset_complexity_profiler.probes import (
    classification_scorer,
    pair_features,
    sts_scorer,
)


def test_id_profile_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 16))
    profile = estimate_id_profile(X)
    assert profile["intrinsic_dim_pca_95"] >= 1
    assert np.isfinite(profile["intrinsic_dim_twonn"])
    assert set(profile) == {"intrinsic_dim_pca_95", "intrinsic_dim_twonn"}


def test_pair_and_sts_probes():
    rng = np.random.default_rng(2)
    u = rng.normal(size=(40, 8))
    v = u + 0.1 * rng.normal(size=(40, 8))
    y = np.array([0] * 20 + [1] * 20)
    X_pair = pair_features(u, v)
    assert X_pair.shape[1] == 24
    acc = classification_scorer(X_pair, y)
    assert 0.0 <= acc <= 1.0

    scores = np.linspace(0, 1, 40)
    X_sts = np.hstack([u, u + 0.05 * rng.normal(size=u.shape)])
    corr = sts_scorer(X_sts, scores)
    assert np.isfinite(corr)
    assert corr >= 0.0


def test_classification_scorer_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = np.array([0] * 45 + [1] * 15)
    first = classification_scorer(X, y)
    second = classification_scorer(X, y)
    assert first == second
    assert 0.0 <= first <= 1.0


def test_sts_scorer_abs_treats_distance_labels_as_signal():
    rng = np.random.default_rng(2)
    u = rng.normal(size=(40, 8))
    scores = np.linspace(0, 1, 40)
    X_sts = np.hstack([u, u + 0.05 * rng.normal(size=u.shape)])
    sim = sts_scorer(X_sts, scores)
    dist = sts_scorer(X_sts, -scores)
    assert sim >= 0.0 and dist >= 0.0
    assert abs(sim - dist) < 1e-12

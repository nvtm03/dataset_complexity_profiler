"""Intrinsic-dimension estimators: PCA-95 and TwoNN (Facco et al.)."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

MIN_TWONN_SAMPLES = 5


def subsample_rows(X: np.ndarray, max_points: int, random_state: int = 42) -> np.ndarray:
    """Deterministic row subsample without replacement (kNN estimators are quadratic)."""
    n_samples = X.shape[0]
    if n_samples <= max_points:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_samples, size=max_points, replace=False)
    return X[idx]


def twonn_id(X: np.ndarray, max_points: int = 2000) -> float:
    """TwoNN (Facco): ID ≈ N / Σ log(r2 / r1). Too few points or a failed kNN → 0.0."""
    n_samples = X.shape[0]
    if n_samples < MIN_TWONN_SAMPLES:
        return 0.0
    X_use = subsample_rows(X, max_points)
    try:
        nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
        nn.fit(X_use)
        distances, _ = nn.kneighbors(X_use)
        r1 = distances[:, 1]
        r2 = distances[:, 2]
        eps = 1e-12
        valid = (r1 > eps) & (r2 > r1)
        if not np.any(valid):
            return 0.0
        mu = r2[valid] / r1[valid]
        denom = float(np.sum(np.log(mu)))
        if denom <= 0.0:
            return 0.0
        return float(np.clip(valid.sum() / denom, 0.0, float(X.shape[1])))
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        logger.debug("TwoNN ID estimation failed: %s", exc)
        return 0.0


def pca_id(X: np.ndarray, target_variance: float = 0.95) -> float:
    """Number of principal components covering ``target_variance`` of variance."""
    X_use = subsample_rows(X, max_points=2000)
    max_components = min(X_use.shape[1], X_use.shape[0] - 1)
    if max_components < 1:
        return 1.0
    pca = PCA(n_components=max_components)
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            pca.fit(X_use)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        if np.isnan(cumulative).all():
            return 1.0
        idx = int(np.searchsorted(cumulative, target_variance, side="left") + 1)
        return float(max(1, min(idx, X.shape[1])))
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.debug("PCA ID estimation failed (likely numerical overflow): %s", exc)
        return 1.0


def estimate_id_profile(X: np.ndarray) -> Dict[str, float]:
    """TwoNN and PCA-95 used as predictive-path meta-features."""
    return {
        "intrinsic_dim_twonn": float(twonn_id(X)),
        "intrinsic_dim_pca_95": float(pca_id(X, 0.95)),
    }

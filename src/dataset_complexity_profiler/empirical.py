"""Empirical PCA search: smallest dim that keeps ≥threshold of full-vector probe quality."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

from .defaults import (
    DEFAULT_QUALITY_THRESHOLD,
    ODD_PAIR_STS_FEATURES_MSG,
    QUALITY_CV_REPEATS,
    check_min_samples_for_recommendation,
    normalize_task_family,
)
from .features import evaluate_quality, prepare_data
from .id_estimators import pca_id
from .probes import (
    check_probe_memory_budget,
    cv_splits,
    get_scorer,
    linear_separability_status,
    require_linear_separability,
    row_cosine,
)

logger = logging.getLogger(__name__)


def pca_intrinsic_dim(X: np.ndarray, target_variance: float = 0.95) -> int:
    """Number of principal components covering ``target_variance`` of variance."""
    return int(pca_id(X, target_variance=target_variance))


def classification_quality_at_dim(
    X: np.ndarray,
    y: np.ndarray,
    dim: int,
    n_repeats: int = QUALITY_CV_REPEATS,
) -> float:
    """Ridge accuracy: PCA fit only on the train fold. Repeated CV for a stable label."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return 0.0
    check_probe_memory_budget(X, y)
    all_scores: List[float] = []
    repeats = max(1, int(n_repeats))
    for seed_offset in range(repeats):
        try:
            splits = cv_splits(X, y, n_splits=3, random_state=42 + seed_offset)
        except ValueError:
            splits = []
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_train)) < 2:
                all_scores.append(0.0)
                continue
            n_comp = min(int(dim), X_train.shape[0] - 1, X_train.shape[1])
            if n_comp < 1:
                continue
            pca = PCA(n_components=n_comp, random_state=42)
            Xtr = pca.fit_transform(X_train)
            Xte = pca.transform(X_test)
            clf = RidgeClassifier(random_state=42)
            clf.fit(Xtr, y_train)
            all_scores.append(float(clf.score(Xte, y_test)))
    if all_scores:
        return float(np.mean(all_scores))
    return 0.0


def sts_quality_at_dim(u: np.ndarray, v: np.ndarray, scores: np.ndarray, dim: int) -> float:
    """PCA on train pairs, |Spearman| on holdout (3 random splits)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n < 2:
        return 0.0

    fold_scores: List[float] = []
    for seed in range(3):
        rng = np.random.default_rng(42 + seed)
        perm = rng.permutation(n)
        cut = max(1, min(n - 1, max(2, int(0.3 * n)) if n >= 4 else 1))
        test = perm[:cut]
        train = perm[cut:]
        if train.size == 0 or test.size == 0:
            continue
        stacked_tr = np.vstack([u[train], v[train]])
        n_comp = min(int(dim), stacked_tr.shape[0] - 1, stacked_tr.shape[1])
        if n_comp < 1:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(stacked_tr)
        sims = row_cosine(pca.transform(u[test]), pca.transform(v[test]))
        y_te = scores[test]
        if np.std(sims) < 1e-12 or np.std(y_te) < 1e-12:
            continue
        corr = spearmanr(sims, y_te).correlation
        if corr is not None and np.isfinite(corr):
            fold_scores.append(abs(float(corr)))
    if fold_scores:
        return float(np.mean(fold_scores))
    return 0.0


def quality_at_dim(X: np.ndarray, y: np.ndarray, dim: int, task_family: str = "single") -> float:
    """Probe quality at PCA-dim. PCA is always fit on train."""
    family = normalize_task_family(task_family)
    if family == "sts":
        if X.shape[1] % 2 != 0:
            raise ValueError(ODD_PAIR_STS_FEATURES_MSG)
        if X.shape[1] < 4:
            return 0.0
        half = X.shape[1] // 2
        return sts_quality_at_dim(X[:, :half], X[:, half:], np.asarray(y, dtype=np.float64), dim)
    return classification_quality_at_dim(X, y, dim)


def candidate_pca_dims(
    effective_max_dim: int,
    min_dim_bound: int = 2,
    step: int = 8,
) -> List[int]:
    """PCA-dim grid. Always includes ``min_dim_bound`` and ``effective_max_dim``."""
    if effective_max_dim < 2:
        return [max(1, effective_max_dim)]
    if effective_max_dim <= min_dim_bound:
        return list(range(max(2, min_dim_bound), effective_max_dim + 1)) or [effective_max_dim]

    low_cap = min(128, effective_max_dim)
    first_step = ((max(min_dim_bound, step) + step - 1) // step) * step
    stepped = list(range(first_step, low_cap + 1, step))
    dims = {int(min_dim_bound), *stepped, int(effective_max_dim)}
    if min_dim_bound <= 4:
        dims.update(d for d in (2, 4, 8) if min_dim_bound <= d <= effective_max_dim)
    if effective_max_dim > low_cap:
        high = np.linspace(low_cap, effective_max_dim, num=16, dtype=int)
        dims.update(int(x) for x in high)
    return sorted(d for d in dims if 2 <= d <= effective_max_dim)


def build_quality_curve(
    X: np.ndarray,
    y: np.ndarray,
    dims: List[int],
    baseline_score: float,
    threshold: float,
    min_dim_bound: int = 2,
    task_family: str = "single",
    stop_at_threshold: bool = False,
) -> Tuple[List[Dict[str, float]], int, float]:
    """Monotone quality-vs-PCA-dim curve. Classification scores are repeated-CV means."""
    safe_max_dim = dims[-1] if dims else 1
    search_dims = [int(d) for d in dims if int(d) >= min(min_dim_bound, safe_max_dim)]
    if not search_dims:
        search_dims = [max(min_dim_bound, int(dims[-1]))] if dims else [max(2, min_dim_bound)]

    family = task_family or "single"
    threshold_value = float(baseline_score) * float(threshold)
    smoothed_curve: List[Dict[str, float]] = []
    best_dim = max(min_dim_bound, search_dims[0])
    best_score = 0.0
    max_seen = 0.0
    n_success = 0

    for dim in search_dims:
        try:
            score = float(quality_at_dim(X, y, int(dim), task_family=family))
            n_success += 1
        except (ValueError, np.linalg.LinAlgError):
            logger.debug("quality_at_dim failed at dim=%s", dim, exc_info=True)
            score = 0.0
        max_seen = max(max_seen, score)
        smoothed_curve.append(
            {
                "dim": int(dim),
                "score": float(score),
                "max_seen": float(max_seen),
                "quality": float(max_seen),
            }
        )
        if max_seen > best_score:
            best_score = max_seen
            best_dim = max(min_dim_bound, int(dim))
        if stop_at_threshold and max_seen >= threshold_value and int(dim) >= min_dim_bound:
            break

    if n_success == 0:
        raise ValueError(
            "All quality_at_dim evaluations failed; cannot recommend a PCA dimension."
        )

    recommended_dim = next(
        (
            max(min_dim_bound, int(entry["dim"]))
            for entry in smoothed_curve
            if entry["max_seen"] >= threshold_value and entry["dim"] >= min_dim_bound
        ),
        max(min_dim_bound, best_dim),
    )
    return smoothed_curve, int(recommended_dim), float(best_score)


def estimate_intrinsic_dim(
    X: Any,
    y: Any,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    max_dim: Optional[int] = None,
    min_dim_bound: int = 2,
    task_family: str = "single",
    stop_at_threshold: bool = False,
    precomputed_baseline: Optional[float] = None,
    precomputed_pca95: Optional[float] = None,
) -> Dict[str, object]:
    """Empirical PCA search: keep ≥ ``quality_threshold`` of full-vector probe quality."""
    task_family = normalize_task_family(task_family)
    X_clean, y_arr = prepare_data(X, y, task_family=task_family)
    check_min_samples_for_recommendation(X_clean.shape[0])
    original_dim = X_clean.shape[1]
    pca_cap = original_dim
    if task_family == "sts" and original_dim >= 4 and original_dim % 2 == 0:
        pca_cap = original_dim // 2
    effective_max_dim = int(min(max_dim or pca_cap, pca_cap, X_clean.shape[0] - 1))
    quality_fn = get_scorer(task_family)

    if precomputed_baseline is not None:
        baseline_score = float(precomputed_baseline)
    else:
        baseline_score = evaluate_quality(X_clean, y_arr, scorer=quality_fn)
    require_linear_separability(
        linear_separability_status(baseline_score, y_arr, task_family=task_family),
    )
    dims = candidate_pca_dims(effective_max_dim, min_dim_bound=min_dim_bound)
    try:
        search_baseline = float(
            quality_at_dim(X_clean, y_arr, int(effective_max_dim), task_family=task_family)
        )
    except (ValueError, np.linalg.LinAlgError):
        search_baseline = float(baseline_score)

    curve, recommended_dim, best_score = build_quality_curve(
        X_clean,
        y_arr,
        dims,
        search_baseline,
        quality_threshold,
        min_dim_bound=min_dim_bound,
        task_family=task_family,
        stop_at_threshold=stop_at_threshold,
    )

    if precomputed_pca95 is not None:
        explained_dim = int(round(float(precomputed_pca95)))
    else:
        explained_dim = pca_intrinsic_dim(X_clean, target_variance=0.95)

    return {
        "original_dim": original_dim,
        "intrinsic_dim_estimate": int(explained_dim),
        "recommended_dim": int(recommended_dim),
        "recommended_threshold": round(search_baseline * quality_threshold, 4),
        "baseline_quality": round(baseline_score, 4),
        "best_dim_quality": round(best_score, 4),
        "quality_threshold": float(quality_threshold),
        "quality_curve": curve,
        "min_dim_bound": int(min_dim_bound),
        "task_family": task_family,
    }

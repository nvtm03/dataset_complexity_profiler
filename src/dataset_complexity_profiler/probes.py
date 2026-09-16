"""Linear probes for empirical search: Ridge accuracy and STS |Spearman|."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .defaults import (
    COLLECT_MIN_LINEAR_KAPPA,
    COLLECT_MIN_STS_ABS_SPEARMAN,
    MAX_PROBE_SAMPLE_CLASS_ELEMENTS,
    ODD_PAIR_STS_FEATURES_MSG,
    normalize_task_family,
)

Scorer = Callable[[np.ndarray, np.ndarray], float]


def majority_chance(y: Sequence[Any]) -> float:
    """Share of the majority class (dummy accuracy)."""
    arr = np.asarray(y)
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    return float(np.max(counts) / arr.size)


def adjusted_accuracy_gain(baseline: float, chance: float) -> float:
    """(acc − majority) / (1 − majority). 0 if the probe is not better than dummy. Not Cohen's kappa."""
    baseline = float(baseline)
    chance = float(np.clip(chance, 0.0, 1.0))
    denom = 1.0 - chance
    if denom <= 1e-12:
        return 1.0 if baseline >= chance - 1e-12 else 0.0
    return float((baseline - chance) / denom)


def check_probe_memory_budget(X: np.ndarray, y: np.ndarray) -> None:
    """Refuse the linear probe when n_samples x n_classes would allocate ~GBs.

    RidgeClassifier one-hot encodes y into an (n_samples, n_classes) matrix; the class
    count is user-controlled: without a budget, e.g. 50k classes x 50k rows is a ~13 GB
    allocation that OOM-kills the host before any separability gate can refuse.
    """
    n_classes = int(np.unique(y).size)
    if X.shape[0] * n_classes > MAX_PROBE_SAMPLE_CLASS_ELEMENTS:
        raise ValueError(
            f"Too many classes for the linear probe: {n_classes} classes x "
            f"{X.shape[0]} rows exceeds the memory budget "
            f"({MAX_PROBE_SAMPLE_CLASS_ELEMENTS} sample-class elements)."
        )


def linear_separability_status(
    baseline: float,
    y: Sequence[Any],
    task_family: str = "single",
    min_kappa: float = COLLECT_MIN_LINEAR_KAPPA,
    min_sts_abs_spearman: float = COLLECT_MIN_STS_ABS_SPEARMAN,
) -> Dict[str, Any]:
    """Whether the linear probe beats dummy enough to write a dim label.

    Classification/pair: adjusted accuracy gain vs majority (not Cohen's kappa, not 1/C).
    STS: |Spearman|.
    """
    baseline_f = float(baseline)
    family = str(task_family or "single")
    if family == "sts":
        value = abs(baseline_f)
        passed = value >= float(min_sts_abs_spearman)
        message = None
        if not passed:
            message = (
                "failed linear separability threshold "
                f"(|Spearman|={value:.3f} < {float(min_sts_abs_spearman):.2f})"
            )
        return {
            "passed": passed,
            "metric": "abs_spearman",
            "value": value,
            "threshold": float(min_sts_abs_spearman),
            "chance": None,
            "baseline": baseline_f,
            "message": message,
        }

    chance = majority_chance(y)
    gain = adjusted_accuracy_gain(baseline_f, chance)
    passed = gain >= float(min_kappa)
    message = None
    if not passed:
        message = (
            "failed linear separability threshold "
            f"(adjusted accuracy gain vs majority={gain:.3f} < {float(min_kappa):.2f}; "
            f"majority chance={chance:.3f}, Ridge={baseline_f:.3f})"
        )
    return {
        "passed": passed,
        "metric": "adjusted_accuracy_gain",
        "value": float(gain),
        "threshold": float(min_kappa),
        "chance": float(chance),
        "baseline": baseline_f,
        "message": message,
    }


def require_linear_separability(
    status: Dict[str, Any],
    *,
    dataset_name: Optional[str] = None,
) -> None:
    """Refuse if the linear probe does not beat dummy: there is nothing to compress."""
    if status.get("passed", True):
        return
    detail = status.get("message") or (
        "linear probe is not better than a dummy baseline"
    )
    prefix = f"{dataset_name}: " if dataset_name else ""
    raise ValueError(
        f"{prefix}Cannot recommend a PCA dimension: {detail}. "
        "The full-vector probe is too close to chance, so keeping 97% of that "
        "score is not a compression recommendation."
    )


def cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = int(min(n_splits, len(y)))
    if n_splits < 2:
        raise ValueError("Need at least 2 samples for CV")
    _, bincount = np.unique(y, return_counts=True)
    if len(bincount) >= 2 and np.min(bincount) >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(cv.split(X, y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(X))


def _ridge_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return 0.0
    check_probe_memory_budget(X, y)
    estimator = RidgeClassifier(random_state=42)
    try:
        splits = cv_splits(X, y, n_splits=3)
    except ValueError:
        splits = None
    if splits:
        scores = []
        for train_idx, test_idx in splits:
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                scores.append(0.0)
                continue
            estimator.fit(X[train_idx], y_train)
            scores.append(float(estimator.score(X[test_idx], y[test_idx])))
        if scores:
            return float(np.mean(scores))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=None
        )
    if len(np.unique(y_train)) < 2:
        return 0.0
    estimator.fit(X_train, y_train)
    return float(estimator.score(X_test, y_test))


def classification_scorer(X: np.ndarray, y: np.ndarray) -> float:
    """Ridge 3-fold accuracy. Also used for pair-clf on [u; v; |u-v|]."""
    return _ridge_accuracy(X, y)


def pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Standard NLI/paraphrase vector: [u; v; |u-v|]."""
    return np.concatenate([u, v, np.abs(u - v)], axis=1)


def row_cosine(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    un = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    un = np.maximum(un, 1e-12)
    return np.sum(u * v, axis=1) / un


def sts_scorer(X: np.ndarray, y: np.ndarray) -> float:
    """|Spearman|(cosine(u, v), gold). Distance labels (negative r) count as signal."""
    if X.shape[1] % 2 != 0:
        raise ValueError(ODD_PAIR_STS_FEATURES_MSG)
    if X.shape[1] < 2:
        return 0.0
    dim = X.shape[1] // 2
    u, v = X[:, :dim], X[:, dim : 2 * dim]
    sims = row_cosine(u, v)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(sims) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0
    corr = spearmanr(sims, y_arr).correlation
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def get_scorer(task_family: str) -> Scorer:
    """Ridge accuracy, except ``sts`` which uses |Spearman| of cosine vs gold."""
    family = normalize_task_family(task_family)
    if family == "sts":
        return sts_scorer
    return classification_scorer

"""Five meta-features consumed by the packaged Random Forest."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .defaults import (
    MAX_PROBE_SAMPLE_CLASS_ELEMENTS,
    ODD_PAIR_STS_FEATURES_MSG,
    STS_QUANTILE_BINS,
    normalize_task_family,
)
from .id_estimators import estimate_id_profile
from .meta_artifact import INTERPRETABLE_FEATURE_NAMES
from .probes import check_probe_memory_budget, classification_scorer, sts_scorer

FeatureMap = Dict[str, float]


def _coerce_label_series(y: Any) -> pd.Series:
    """Flatten labels from DataFrame / Series / ndarray / nested lists to a 1-d Series."""
    if isinstance(y, pd.DataFrame):
        y_series = y.iloc[:, 0]
    elif isinstance(y, pd.Series):
        y_series = y
    else:
        if isinstance(y, np.ndarray):
            y_arr_raw = y.ravel()
        else:
            y_arr_raw = np.array(y, dtype=object).ravel()
        y_series = pd.Series(y_arr_raw)
    if y_series.dtype.kind in "OSU":
        y_series = y_series.replace({"nan": np.nan, "NaN": np.nan})
    return y_series


def prepare_data(
    X: Any,
    y: Any,
    task_family: str = "single",
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean NaNs in ``X`` and normalise labels for the probe family."""
    X_arr = np.asarray(X)
    if X_arr.dtype.kind == "c":
        raise ValueError("Complex embeddings are not supported.")
    if X_arr.dtype != np.float64:
        X_arr = X_arr.astype(np.float64)
    if X_arr.ndim not in (1, 2):
        raise ValueError("X must be 1D or 2D")
    y_series = _coerce_label_series(y)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.shape[0] == 0:
        raise ValueError("Input dataset is empty (0 samples).")
    if X_arr.shape[0] != len(y_series):
        raise ValueError(
            f"X and y length mismatch: {X_arr.shape[0]} vs {len(y_series)}"
        )

    family = normalize_task_family(task_family)
    if family in {"pair", "sts"} and X_arr.shape[1] % 2 != 0:
        raise ValueError(ODD_PAIR_STS_FEATURES_MSG)

    present = y_series.notna().to_numpy()
    if not present.all():
        n_drop = int((~present).sum())
        warnings.warn(
            f"Dropped {n_drop} rows with missing labels; they are not treated as a class.",
            UserWarning,
            stacklevel=2,
        )
        X_arr = X_arr[present]
        y_series = y_series.iloc[np.flatnonzero(present)]
        if X_arr.shape[0] == 0:
            raise ValueError("Input dataset is empty after dropping missing labels.")

    y_arr = y_series.to_numpy()

    if np.isfinite(X_arr).all():
        X_clean = X_arr
    else:
        X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if family == "sts":
        y_out = np.asarray(y_arr, dtype=np.float64)
    elif y_arr.dtype.kind in "iub":
        y_out = np.asarray(y_arr, dtype=int)
    elif y_arr.dtype.kind == "f":
        rounded = np.round(y_arr.astype(np.float64))
        if not np.isfinite(rounded).all() or (np.abs(rounded) > 2**53).any():
            y_out = pd.factorize(y_arr)[0]
        elif np.allclose(y_arr.astype(np.float64), rounded, equal_nan=True):
            y_out = np.asarray(rounded, dtype=int)
        else:
            y_out = pd.factorize(y_arr)[0]
    else:
        y_out = pd.factorize(y_arr)[0]

    if y_out.size and family != "sts" and np.min(y_out) < 0:
        y_out = pd.factorize(y_out)[0]

    if np.unique(y_out).size < 2:
        raise ValueError("Dataset must contain at least 2 classes.")
    return X_clean, y_out


def evaluate_quality(
    X: np.ndarray,
    y: np.ndarray,
    scorer: Optional[object] = None,
) -> float:
    """Full-vector probe score. Does not fall back to Ridge (unsafe for STS y)."""
    if X.shape[1] == 0:
        return 0.0
    check_probe_memory_budget(X, y)
    quality_fn = scorer or classification_scorer
    try:
        return float(quality_fn(X, y))
    except (ValueError, np.linalg.LinAlgError):
        return 0.0


def id_profile_from_named(named: Mapping[str, float]) -> Dict[str, float]:
    """Rebuild the ID profile from already computed features (no extra kNN/PCA)."""
    return {
        "intrinsic_dim_twonn": float(named.get("intrinsic_dim_twonn") or 0.0),
        "intrinsic_dim_pca_95": float(named.get("intrinsic_dim_pca_95") or 0.0),
    }


def named_feature_vector(
    features: Union[Mapping[str, float], np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    names = list(feature_names) if feature_names is not None else list(INTERPRETABLE_FEATURE_NAMES)
    if isinstance(features, Mapping):
        missing = [name for name in names if name not in features]
        if missing:
            raise ValueError(f"Feature name mismatch: missing {missing}")
        values = np.array([float(features[name]) for name in names], dtype=np.float64)
    else:
        values = np.asarray(features, dtype=np.float64).ravel()
        if values.size != len(names):
            raise ValueError(
                f"Feature vector length mismatch: {values.size} != {len(names)} names"
            )
    return names, np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _build_meta_feature_vector_prepared(
    X: Any,
    y: Any,
    task_family: str = "single",
) -> FeatureMap:
    """Five contract features; ``X`` and ``y`` must already be prepared."""
    family = normalize_task_family(task_family)
    X_clean = np.asarray(X)
    y_prepared = np.asarray(y)
    if family == "sts":
        y_scores = np.asarray(y_prepared, dtype=np.float64)
        bins = np.unique(np.quantile(y_scores, STS_QUANTILE_BINS))
        if bins.size < 2:
            y_arr = pd.factorize(y_scores)[0]
        else:
            y_arr = np.digitize(y_scores, bins=bins)
        baseline_quality = float(sts_scorer(X_clean, y_scores))
    else:
        y_arr = y_prepared
        baseline_quality = float(evaluate_quality(X_clean, y_arr))

    class_count = np.unique(y_arr).size

    id_profile = estimate_id_profile(X_clean)

    mean_centroid_distance = 0.0
    if class_count > 1 and X_clean.shape[0] > class_count:
        centroid_elements = class_count * class_count
        if centroid_elements > MAX_PROBE_SAMPLE_CLASS_ELEMENTS:
            warnings.warn(
                f"Skipping centroid distance: {class_count} classes exceeds memory budget "
                f"({MAX_PROBE_SAMPLE_CLASS_ELEMENTS} pairwise elements)",
                UserWarning,
                stacklevel=2,
            )
        else:
            _classes, inverse_indices = np.unique(y_arr, return_inverse=True)
            counts = np.bincount(inverse_indices, minlength=len(_classes))[:, None]
            sums = np.zeros((len(_classes), X_clean.shape[1]), dtype=np.float64)
            np.add.at(sums, inverse_indices, X_clean)
            class_centroids = sums / np.maximum(counts, 1)
            pairwise = cdist(class_centroids, class_centroids, metric="cosine")
            upper = np.asarray(pairwise[np.triu_indices_from(pairwise, k=1)], dtype=np.float64)
            if upper.size:
                upper[np.isnan(upper)] = 1.0
                mean_centroid_distance = float(np.mean(upper))

    raw: FeatureMap = {
        "class_count": float(class_count),
        "baseline_quality": float(baseline_quality),
        "mean_centroid_cosine_distance": float(mean_centroid_distance),
        "intrinsic_dim_twonn": float(id_profile["intrinsic_dim_twonn"]),
        "intrinsic_dim_pca_95": float(id_profile["intrinsic_dim_pca_95"]),
    }
    features: FeatureMap = {}
    for name in INTERPRETABLE_FEATURE_NAMES:
        value = float(raw[name])
        features[name] = value if np.isfinite(value) else 0.0
    return features


def build_meta_feature_vector(
    X: Any,
    y: Any,
    task_family: str = "single",
) -> FeatureMap:
    """Return the five contract features in ``INTERPRETABLE_FEATURE_NAMES`` order."""
    family = normalize_task_family(task_family)
    X_clean, y_prepared = prepare_data(X, y, task_family=family)
    return _build_meta_feature_vector_prepared(X_clean, y_prepared, task_family=family)

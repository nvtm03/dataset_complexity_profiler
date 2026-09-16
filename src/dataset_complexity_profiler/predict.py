"""Packaged meta-model: load artifact and predict a PCA-dimension bucket."""

from __future__ import annotations

import logging
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .defaults import (
    DIM_BUCKETS,
    PACKAGED_META_MODEL_NAME,
    PREDICT_PROBA_CDF_THRESHOLD,
    check_min_samples_for_recommendation,
    feasible_dim_bucket,
    normalize_task_family,
    warn_if_not_bucket,
)
from .features import (
    _build_meta_feature_vector_prepared,
    named_feature_vector,
    prepare_data,
)
from .meta_artifact import (
    INTERPRETABLE_FEATURE_NAMES,
    align_features_by_name,
    is_meta_artifact,
    load_artifact,
    warn_sklearn_mismatch,
)
from .probes import linear_separability_status, require_linear_separability

logger = logging.getLogger(__name__)


def load_meta_artifact(model_path: str) -> Tuple[Any, List[str]]:
    """Return ``(pipeline, feature_names)``. Names are empty for a bare estimator."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Meta-model file not found: {model_path}")
    loaded = load_artifact(path)
    if is_meta_artifact(loaded):
        warn_sklearn_mismatch(loaded)
        return loaded["pipeline"], list(loaded.get("feature_names") or [])
    elif isinstance(loaded, dict):
        raise ValueError("Artifact is a dictionary but not a valid v2 meta-artifact.")
    else:
        return loaded, []


def is_recoverable_predict_error(exc: BaseException) -> bool:
    """True if predict failed for a missing/unfitted model or feature-name mismatch."""
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc).lower()
    return msg.startswith(
        (
            "meta model is not fitted",
            "feature name mismatch",
            "feature vector is all zeros",
        )
    ) or (msg.startswith("this ") and "not fitted" in msg)


def conservative_dim_from_proba(
    probabilities: np.ndarray,
    classes: np.ndarray,
    cdf_threshold: float = PREDICT_PROBA_CDF_THRESHOLD,
) -> int:
    """Smallest bucket whose cumulative P(dim ≤ bucket) is ≥ ``cdf_threshold``.

    Classes are ordered by increasing dimension: overshooting (extra components)
    is cheaper than undershooting and collapsing the linear probe.
    """
    classes_arr = np.asarray(classes)
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    if classes_arr.size == 0 or probs.size == 0:
        raise ValueError("Cannot pick a bucket from empty predict_proba output.")
    if probs.size != classes_arr.size:
        raise ValueError(
            f"predict_proba length {probs.size} != n_classes {classes_arr.size}"
        )
    order = np.argsort(classes_arr.astype(int), kind="stable")
    ordered_classes = classes_arr[order]
    cdf = np.cumsum(np.maximum(probs[order], 0.0))
    total = float(cdf[-1]) if cdf.size else 0.0
    if np.isnan(probs).any() or np.isnan(total) or total <= 0.0:
        return int(ordered_classes[-1])
    cdf = cdf / total
    idx = int(np.searchsorted(cdf, float(cdf_threshold), side="left"))
    idx = min(max(idx, 0), int(ordered_classes.size) - 1)
    return int(ordered_classes[idx])


def load_default_meta_artifact(model_path: Optional[str] = None) -> Optional[Tuple[Any, List[str]]]:
    """Load packaged ``meta_model.skops``, or ``model_path``. Missing file → ``None``."""
    if model_path is not None:
        return load_meta_artifact(model_path)
    resource = files(__package__).joinpath(PACKAGED_META_MODEL_NAME)
    try:
        with as_file(resource) as path:
            if not Path(path).is_file():
                logger.warning("Packaged %s is missing", PACKAGED_META_MODEL_NAME)
                return None
            return load_meta_artifact(str(path))
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Packaged %s is missing: %s", PACKAGED_META_MODEL_NAME, exc)
        return None


def predict_embedding_dim(
    X: Any,
    y: Any,
    meta_model: Any,
    *,
    meta_feature_names: Optional[Sequence[str]] = None,
    task_family: str = "single",
    features: Optional[Union[Mapping[str, float], np.ndarray]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> int:
    """Predict a PCA-dim bucket (CDF safety margin), then clip to matrix rank."""
    if meta_model is None:
        raise ValueError("Meta model is not fitted yet. Call load_meta_model() first.")

    family = normalize_task_family(task_family)
    stored_names = list(meta_feature_names or [])
    if features is None:
        X_clean, y_arr = prepare_data(X, y, task_family=family)
        feat_map = _build_meta_feature_vector_prepared(X_clean, y_arr, task_family=family)
        names, feat_vec = named_feature_vector(feat_map)
        X_for_shape = X_clean
        y_for_gate = y_arr
    else:
        fallback = list(stored_names or INTERPRETABLE_FEATURE_NAMES)
        names, feat_vec = named_feature_vector(features, feature_names or fallback)
        X_for_shape = np.asarray(X)
        if X_for_shape.ndim not in (1, 2):
            raise ValueError("X must be 1D or 2D")
        y_for_gate = np.asarray(y).ravel()
        if X_for_shape.shape[0] != len(y_for_gate):
            raise ValueError("X and y length mismatch")

    X_arr = np.asarray(X_for_shape)
    n_samples = X_arr.shape[0]
    n_features = 1 if X_arr.ndim == 1 else X_arr.shape[1]
    check_min_samples_for_recommendation(n_samples)
    if family == "sts" and n_features >= 2:
        n_features = n_features // 2

    feat_mat = np.asarray(feat_vec, dtype=np.float64).reshape(1, -1)
    expected_names = list(stored_names or INTERPRETABLE_FEATURE_NAMES)
    feat_mat = align_features_by_name(names, feat_mat.ravel(), expected_names).reshape(1, -1)
    if not np.isfinite(feat_mat).all() or np.allclose(feat_mat, 0.0):
        raise ValueError(
            "Feature vector is all zeros or non-finite (feature mismatch or empty input). "
            "Refusing to predict a dummy bucket."
        )
    name_to_val = {name: float(val) for name, val in zip(expected_names, feat_mat.ravel())}
    baseline = float(name_to_val.get("baseline_quality") or 0.0)
    require_linear_separability(
        linear_separability_status(baseline, y_for_gate, task_family=family),
    )

    if hasattr(meta_model, "predict_proba"):
        probabilities = np.asarray(meta_model.predict_proba(feat_mat)[0], dtype=np.float64)
        classes = np.asarray(getattr(meta_model, "classes_"))
        predicted = conservative_dim_from_proba(probabilities, classes)
        logger.debug(
            "Probabilities=%s CDF-threshold=%.2f → dim=%s",
            np.round(probabilities, 3),
            PREDICT_PROBA_CDF_THRESHOLD,
            predicted,
        )
    else:
        predicted = int(meta_model.predict(feat_mat)[0])
    clipped = feasible_dim_bucket(predicted, n_features, n_samples)
    if clipped != predicted:
        logger.debug(
            "Clipped predicted dim %s → %s for matrix %s×%s (buckets=%s)",
            predicted,
            clipped,
            n_samples,
            n_features,
            DIM_BUCKETS,
        )
    warn_if_not_bucket(clipped)
    return clipped

"""v2 artifact: dict with a bare RF under the key ``pipeline`` (skops, not pickle)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

META_ARTIFACT_VERSION = 2

SKOPS_TRUSTED_TYPES: FrozenSet[str] = frozenset(
    {
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
    }
)

INTERPRETABLE_FEATURE_NAMES: Tuple[str, ...] = (
    "class_count",
    "baseline_quality",
    "mean_centroid_cosine_distance",
    "intrinsic_dim_twonn",
    "intrinsic_dim_pca_95",
)

TARGET_COLUMN = "recommended_dim"
ID_COLUMN = "dataset_name"

LINEAR_SEP_METRIC_COLUMN = "linear_sep_metric"
LINEAR_SEP_VALUE_COLUMN = "linear_sep_value"
LINEAR_SEP_THRESHOLD_COLUMN = "linear_sep_threshold"
LINEAR_SEP_COLUMNS: List[str] = [
    LINEAR_SEP_METRIC_COLUMN,
    LINEAR_SEP_VALUE_COLUMN,
    LINEAR_SEP_THRESHOLD_COLUMN,
]

META_CSV_COLUMNS: List[str] = [
    ID_COLUMN,
    TARGET_COLUMN,
    "sample_count",
    "original_dim",
    *INTERPRETABLE_FEATURE_NAMES,
    "minority_fraction",
    "task_family",
    "true_intrinsic_dim",
    "is_synthetic",
    "language",
    *LINEAR_SEP_COLUMNS,
]


def separability_row_values(status: Optional[Mapping[str, Any]]) -> List[Any]:
    """``[metric, value, threshold]``; empty strings if the gate was not measured."""
    if not status:
        return ["", "", ""]
    return [
        str(status.get("metric") or ""),
        float(status.get("value") or 0.0),
        float(status.get("threshold") or 0.0),
    ]


def compact_feature_columns(columns: List[str]) -> List[str]:
    """The five contract features in order, skipping names absent from ``columns``."""
    available = set(columns)
    return [name for name in INTERPRETABLE_FEATURE_NAMES if name in available]


def is_meta_artifact(obj: Any) -> bool:
    """True if ``obj`` is a v2 dict, not a bare estimator."""
    return isinstance(obj, dict) and obj.get("version") == META_ARTIFACT_VERSION and "pipeline" in obj


def _sklearn_version() -> Optional[str]:
    try:
        import sklearn

        return str(sklearn.__version__)
    except ImportError:
        return None


def load_artifact(path: Union[str, Path]) -> Any:
    """Load a skops artifact safely. Normalizes any corruption to ValueError."""
    import json
    import struct
    import zipfile
    import zlib

    import skops.io as sio
    from sklearn.ensemble import RandomForestClassifier

    path = Path(path)
    try:
        unknown = sio.get_untrusted_types(file=path)
        extra = sorted(set(unknown) - SKOPS_TRUSTED_TYPES)
        if extra:
            raise ValueError(
                f"Refusing to load meta-model {path}: untrusted types {extra}. "
                "This is not the Dataset Complexity Profiler artifact."
            )
        loaded = sio.load(path, trusted=list(SKOPS_TRUSTED_TYPES))
    except (
        json.JSONDecodeError,
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        AttributeError,
        RuntimeError,
        IndexError,
        struct.error,
        zlib.error,
        EOFError,
    ) as exc:
        raise ValueError(f"Meta-model artifact {path} is corrupted: {exc}") from exc
    except (ValueError, OSError):
        raise

    pipeline = loaded.get("pipeline", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(pipeline, RandomForestClassifier):
        raise ValueError("Artifact pipeline is not a RandomForestClassifier.")
    return loaded


def warn_sklearn_mismatch(artifact: Mapping[str, Any]) -> None:
    stored = artifact.get("sklearn_version")
    if not stored:
        return
    running = _sklearn_version()
    if running is None or str(stored) == running:
        return
    logger.warning(
        "Meta-model artifact was saved with sklearn %s, but this process has %s. "
        "Reload after a matching install or retrain if predictions look off.",
        stored,
        running,
    )


def wrap_artifact(
    pipeline: BaseEstimator,
    *,
    feature_names: List[str],
) -> Dict[str, Any]:
    return {
        "version": META_ARTIFACT_VERSION,
        "pipeline": pipeline,
        "feature_names": list(feature_names),
        "sklearn_version": _sklearn_version(),
    }


def save_artifact(artifact: Dict[str, Any], path: Union[str, Path]) -> None:
    import skops.io as sio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(artifact, path)


def align_features_by_name(
    feature_names: List[str],
    features: np.ndarray,
    expected_names: List[str],
) -> np.ndarray:
    """Reorder ``features`` to ``expected_names``. Missing names raise; they are not zero-filled."""
    values = np.asarray(features, dtype=np.float64).ravel()
    mapping = {name: values[i] for i, name in enumerate(feature_names) if i < values.size}
    missing = [name for name in expected_names if name not in mapping]
    if missing:
        raise ValueError(
            f"Feature name mismatch: missing {missing}. "
            "Refusing to fill missing meta-features with zeros."
        )
    return np.array([mapping[name] for name in expected_names], dtype=np.float64)

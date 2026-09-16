"""Shared library constants."""

from __future__ import annotations

import warnings

DEFAULT_EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QUALITY_THRESHOLD = 0.97
QUALITY_CV_REPEATS = 3
PREDICT_PROBA_CDF_THRESHOLD = 0.80

DIM_BUCKETS = (4, 8, 16, 32, 64, 128, 256)
DIM_BUCKET_BINS = (0, 4, 8, 16, 32, 64, 128, 256, 100000)

MIN_SAMPLES_FOR_DIM_RECOMMENDATION = 30

TASK_FAMILIES = ("single", "pair", "sts")
TASK_FAMILY_ALIASES = {"classification": "single"}

COLLECT_MIN_LINEAR_KAPPA = 0.10
COLLECT_MIN_STS_ABS_SPEARMAN = 0.15

MAX_PROBE_SAMPLE_CLASS_ELEMENTS = 200_000_000

STS_QUANTILE_BINS = [0.25, 0.5, 0.75]

PACKAGED_META_MODEL_NAME = "meta_model.skops"

ODD_PAIR_STS_FEATURES_MSG = (
    "Pair/STS tasks require an even number of features (embeddings for text A and text B)."
)


def normalize_task_family(task_family: str) -> str:
    """Map ``classification`` to ``single``. Unknown names raise ``ValueError``."""
    family = str(task_family).strip().lower()
    family = TASK_FAMILY_ALIASES.get(family, family)
    if family not in TASK_FAMILIES:
        raise ValueError(
            f"Unknown task_family {task_family!r}. "
            f"Expected one of {TASK_FAMILIES} (alias: classification → single)."
        )
    return family


def pca_rank_cap(n_features: int, n_samples: int) -> int:
    """Maximum PCA ``n_components`` on a centered ``(n_samples, n_features)`` matrix."""
    return max(1, min(n_features, max(n_samples - 1, 1)))


def feasible_dim_bucket(
    predicted: int,
    n_features: int,
    n_samples: int,
    buckets: tuple[int, ...] = DIM_BUCKETS,
) -> int:
    """Nearest bucket, then clip to ``min(n_features, n_samples-1)``. Rank < 4 → the cap itself."""
    raw = int(predicted)
    if raw not in buckets:
        raw = min(buckets, key=lambda bucket: (abs(bucket - raw), bucket))
    cap = pca_rank_cap(n_features, n_samples)
    target = min(raw, cap)
    allowed = [bucket for bucket in buckets if bucket <= target]
    return allowed[-1] if allowed else cap


def dim_is_bucket(dim: int, buckets: tuple[int, ...] = DIM_BUCKETS) -> bool:
    return dim in buckets


def check_min_samples_for_recommendation(n_samples: int) -> None:
    if n_samples < MIN_SAMPLES_FOR_DIM_RECOMMENDATION:
        raise ValueError(
            f"Need at least {MIN_SAMPLES_FOR_DIM_RECOMMENDATION} samples to recommend "
            f"a PCA dimension (got {n_samples}). With fewer rows the PCA rank cap is 1–3, "
            "which is not a meta-model bucket and is not an interpretable recommendation."
        )


def warn_if_not_bucket(dim: int, buckets: tuple[int, ...] = DIM_BUCKETS) -> None:
    """Warn when clip left the bucket grid (matrix rank < 4)."""
    if dim_is_bucket(dim, buckets):
        return
    warnings.warn(
        f"Recommended dimension {int(dim)} is not a standard bucket {tuple(buckets)}; "
        "it is an exact PCA rank bound, not a meta-model bucket. "
        f"Need at least {MIN_SAMPLES_FOR_DIM_RECOMMENDATION} samples and rank ≥ 4 "
        "for a bucketed recommendation.",
        UserWarning,
    )

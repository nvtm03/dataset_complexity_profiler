"""Dataset Complexity Profiler: profile a labeled dataset and recommend a compact PCA dim."""

import importlib.metadata

from .complexity_report import interpret_id_profile
from .defaults import DEFAULT_EMBEDDER_NAME
from .meta_artifact import INTERPRETABLE_FEATURE_NAMES
from .profiler import DatasetProfiler

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "DEFAULT_EMBEDDER_NAME",
    "DatasetProfiler",
    "INTERPRETABLE_FEATURE_NAMES",
    "__version__",
    "interpret_id_profile",
]

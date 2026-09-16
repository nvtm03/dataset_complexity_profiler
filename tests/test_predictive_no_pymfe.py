"""Predictive path must not import PyMFE at module load or at predict time."""

from __future__ import annotations

import sys

import numpy as np


def test_profiler_module_does_not_import_pymfe():
    for key in list(sys.modules):
        if key == "pymfe" or key.startswith("pymfe."):
            del sys.modules[key]
    import importlib

    import dataset_complexity_profiler.profiler as profiler_mod

    importlib.reload(profiler_mod)
    assert "pymfe" not in sys.modules
    assert "pymfe.mfe" not in sys.modules


def test_predict_embedding_dim_does_not_import_pymfe():
    for key in list(sys.modules):
        if key == "pymfe" or key.startswith("pymfe."):
            del sys.modules[key]

    from dataset_complexity_profiler import DatasetProfiler

    profiler = DatasetProfiler(auto_load_meta_model=True)
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(0.0, 0.3, size=(20, 24)),
            rng.normal(4.0, 0.3, size=(20, 24)),
        ]
    )
    y = np.array([0] * 20 + [1] * 20)
    dim = profiler.predict_embedding_dim(X, y)
    assert isinstance(dim, int)
    assert dim in {4, 8, 16, 32}
    assert dim <= 24
    assert "pymfe" not in sys.modules
    assert "pymfe.mfe" not in sys.modules

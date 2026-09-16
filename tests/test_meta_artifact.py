from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from dataset_complexity_profiler import INTERPRETABLE_FEATURE_NAMES, DatasetProfiler
from dataset_complexity_profiler.defaults import PACKAGED_META_MODEL_NAME
from dataset_complexity_profiler.meta_artifact import (
    align_features_by_name,
    load_artifact,
    save_artifact,
    wrap_artifact,
)
from dataset_complexity_profiler.predict import load_meta_artifact


def _two_blobs(n=40, d=16, seed=0, sep=4.0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X = np.vstack(
        [
            rng.normal(0.0, 0.4, size=(n0, d)),
            rng.normal(sep, 0.4, size=(n - n0, d)),
        ]
    )
    y = np.array([0] * n0 + [1] * (n - n0))
    return X, y


def _corrupt_artifacts(tmp_path):
    import json
    import pickle
    import zipfile

    good = tmp_path / "good.skops"
    clf = RandomForestClassifier(n_estimators=3, random_state=0)
    X, y = _two_blobs(n=20, d=5)
    clf.fit(X, y)
    save_artifact(wrap_artifact(clf, feature_names=["a", "b", "c", "d", "e"]), good)
    blob = good.read_bytes()
    flipped = bytearray(blob)
    for i in range(len(flipped) // 3, len(flipped) // 3 + 40):
        flipped[i] ^= 0xFF

    cases = {}

    def raw(name, payload):
        path = tmp_path / f"{name}.skops"
        path.write_bytes(payload)
        cases[name] = path

    def zipped(name, schema=None, extra=None):
        path = tmp_path / f"{name}.skops"
        with zipfile.ZipFile(path, "w") as archive:
            if schema is not None:
                archive.writestr("schema.json", schema)
            if extra is not None:
                archive.writestr("hello.txt", extra)
        cases[name] = path

    raw("empty", b"")
    raw("garbage", b"\x00\xff" * 500)
    raw("truncated", blob[: len(blob) // 2])
    raw("bitflip", bytes(flipped))
    raw("pickle", pickle.dumps({"a": 1}))
    zipped("no_schema", extra="hi")
    zipped("bad_json", schema="{not json")
    zipped("empty_schema", schema="{}")
    for label, body in [("list", "[]"), ("null", "null"), ("int", "5"), ("str", '"x"')]:
        zipped(f"json_{label}", schema=body)
    zipped(
        "wrong_field_types",
        schema=json.dumps(
            {
                "__class__": "dict",
                "__module__": "builtins",
                "__loader__": "dict",
                "content": {"key_types": 5, "content": None},
                "protocol": 0,
                "_skops_version": "0.14.0",
            }
        ),
    )
    return cases


def test_interpretable_feature_names_order():
    assert INTERPRETABLE_FEATURE_NAMES == (
        "class_count",
        "baseline_quality",
        "mean_centroid_cosine_distance",
        "intrinsic_dim_twonn",
        "intrinsic_dim_pca_95",
    )
    assert len(INTERPRETABLE_FEATURE_NAMES) == 5


def test_align_features_by_name():
    names = ["a", "b", "c"]
    values = np.array([1.0, 2.0, 3.0])
    aligned = align_features_by_name(names, values, ["c", "a", "b"])
    assert aligned.tolist() == [3.0, 1.0, 2.0]


def test_align_features_by_name_rejects_missing():
    with pytest.raises(ValueError, match="mismatch"):
        align_features_by_name(["a", "b"], np.array([1.0, 2.0]), ["b", "c"])


def test_load_artifact_normalizes_every_corruption_to_value_error(tmp_path):
    for name, path in _corrupt_artifacts(tmp_path).items():
        with pytest.raises(ValueError):
            load_artifact(path)
        assert path.exists(), name


def test_corrupt_packaged_artifact_does_not_crash_profiler_init(tmp_path, monkeypatch):
    for name, path in _corrupt_artifacts(tmp_path).items():
        monkeypatch.setattr(
            "dataset_complexity_profiler.profiler.load_default_meta_artifact",
            lambda _requested=None, _path=path: load_meta_artifact(str(_path)),
        )
        profiler = DatasetProfiler()
        assert profiler.is_fitted is False, name
        assert profiler.meta_model is None, name


def test_explicit_corrupt_meta_model_path_raises_value_error(tmp_path):
    for name, path in _corrupt_artifacts(tmp_path).items():
        with pytest.raises(ValueError):
            DatasetProfiler(auto_load_meta_model=True, meta_model_path=str(path))


def test_packaged_artifact_was_saved_with_sklearn_1_6():
    path = Path(__file__).resolve().parents[1] / "src/dataset_complexity_profiler" / PACKAGED_META_MODEL_NAME
    artifact = load_artifact(path)
    assert str(artifact["sklearn_version"]).startswith("1.6")
    estimator = artifact["pipeline"]
    assert isinstance(estimator, RandomForestClassifier)
    assert not hasattr(estimator, "named_steps")

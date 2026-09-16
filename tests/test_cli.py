from dataset_complexity_profiler.cli import build_parser


def test_cli_parser_analyze():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze",
            "--csv",
            "dummy.csv",
            "--limit",
            "10",
            "--no-meta-model",
            "--task",
            "pair",
            "--text-col-b",
            "text2",
        ]
    )
    assert args.command == "analyze"
    assert args.csv == "dummy.csv"
    assert args.limit == 10
    assert args.no_meta_model is True
    assert args.task == "pair"
    assert args.text_col_b == "text2"
    assert args.trust_remote_code is False


def test_cli_limit_defaults_to_full_file():
    parser = build_parser()
    args = parser.parse_args(["analyze", "--csv", "dummy.csv"])
    assert args.limit is None
    parser = build_parser()
    args = parser.parse_args(["analyze", "--csv", "dummy.csv", "--empirical"])
    assert args.empirical is True
    args = parser.parse_args(["analyze", "--hf", "org/ds", "--trust-remote-code"])
    assert args.trust_remote_code is True


def test_cli_does_not_swallow_index_error(monkeypatch):
    import pytest

    from dataset_complexity_profiler.cli import cmd_analyze

    parser = build_parser()
    args = parser.parse_args(["analyze", "--csv", "dummy.csv", "--no-meta-model"])

    def fake_profiler(*_a, **_k):
        raise IndexError("index 3 is out of bounds")

    monkeypatch.setattr("dataset_complexity_profiler.DatasetProfiler", fake_profiler)
    with pytest.raises(IndexError, match="index 3"):
        cmd_analyze(args)


def test_cli_user_errors_include_io_and_parse_failures():
    from pandas.errors import ParserError

    from dataset_complexity_profiler.cli import _CLI_USER_ERRORS

    for exc_type in (FileNotFoundError, KeyError, OSError, ParserError):
        assert exc_type in _CLI_USER_ERRORS


def test_read_csv_uses_only_requested_columns_and_limit(tmp_path, monkeypatch):
    import pandas as pd

    from dataset_complexity_profiler.cli import _read_csv_texts_labels

    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "noise": range(6),
            "text": ["a", "b", " ", "c", "d", "e"],
            "label": [0, 1, 1, 0, 1, 0],
            "feat": [9.0] * 6,
        }
    ).to_csv(path, index=False)

    calls = []
    real_read_csv = pd.read_csv

    def spy_read_csv(*args, **kwargs):
        calls.append(kwargs)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)

    texts, labels, texts_b = _read_csv_texts_labels(path, "text", "label", limit=2)
    assert texts == ["a", "b"]
    assert labels == [0, 1]
    assert texts_b is None
    data_calls = [c for c in calls if "usecols" in c]
    assert data_calls
    assert data_calls[0]["usecols"] == ["text", "label"]
    assert data_calls[0]["chunksize"] == 4096


def test_load_hf_explains_trust_remote_code(monkeypatch):
    import sys
    import types

    import pytest

    from dataset_complexity_profiler.cli import _load_hf

    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*_args, **_kwargs):
        raise ValueError("This dataset requires trust_remote_code=True")

    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(ValueError, match="Pass --trust-remote-code"):
        _load_hf("org/custom", "train", "text", "label", None, None)


def test_load_hf_passes_trust_remote_code(monkeypatch):
    import sys
    import types

    from dataset_complexity_profiler.cli import _load_hf

    captured = {}

    class FakeDS:
        def filter(self, _fn):
            return self

        def __len__(self):
            return 2

        def __getitem__(self, key):
            if key == "text":
                return ["hello", "world"]
            if key == "label":
                return [0, 1]
            raise KeyError(key)

    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*_args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeDS()

    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    texts, labels, texts_b = _load_hf(
        "org/custom",
        "train",
        "text",
        "label",
        None,
        None,
        trust_remote_code=True,
    )
    assert texts == ["hello", "world"]
    assert labels == [0, 1]
    assert texts_b is None
    assert captured["kwargs"]["split"] == "train"
    assert captured["kwargs"]["trust_remote_code"] is True

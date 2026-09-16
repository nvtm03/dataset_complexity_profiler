from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_collect_module():
    path = Path(__file__).resolve().parents[1] / "research" / "collect_training_extra.py"
    spec = importlib.util.spec_from_file_location("collect_training_extra", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_skips_sources_already_in_selection_csv():
    tmm = _load_collect_module()
    existing = tmm.load_existing_keys(Path("research/feature_selection.csv"))
    assert tmm.is_already_collected(
        {"name": "IMDb (Binary)", "hf": "imdb", "config": None},
        existing,
    )
    assert tmm.is_already_collected(
        {"name": "AG News (4 classes)", "hf": "fancyzhx/ag_news", "config": None},
        existing,
    )
    planned_names = {spec["name"] for spec in tmm.planned_catalog(existing, only="")}
    assert "IMDb (Binary)" not in planned_names
    assert "XNLI English" not in planned_names
    assert "GLUE CoLA" not in planned_names
    assert "Financial PhraseBank" in planned_names
    assert "TweetEval irony" in planned_names


def test_keeps_language_variants_and_new_domains():
    tmm = _load_collect_module()
    existing = tmm.load_existing_keys(Path("research/feature_selection.csv"))
    planned_names = {spec["name"] for spec in tmm.planned_catalog(existing, only="")}
    assert len(tmm.CATALOG) >= 80
    assert len(planned_names) >= 70
    assert "Amazon Reviews DE" in planned_names
    assert "XNLI French" in planned_names
    assert "PAWS-X German" not in planned_names
    assert "SciCite" in planned_names
    assert "MTOP Domain" in planned_names
    assert "RuReviews" in planned_names


class _FakeTable:
    def __init__(self, rows):
        self._rows = list(rows)
        self.column_names = list(self._rows[0].keys()) if self._rows else ["text", "label"]

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def shuffle(self, seed=None):
        return self


def _toy_spec():
    return {
        "name": "Toy",
        "hf": "acme/toy",
        "text": "text",
        "label": "label",
        "family": "single",
    }


def test_load_hf_does_not_retry_trust_remote_code_on_network_error(monkeypatch):
    tmm = _load_collect_module()
    calls = []
    fake = types.ModuleType("datasets")

    def load_dataset(*_args, **kwargs):
        calls.append(dict(kwargs))
        raise ConnectionError("HTTPSConnectionPool timed out")

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    with pytest.raises(RuntimeError, match="timed out"):
        tmm.load_hf_table(_toy_spec(), limit=80)
    assert calls == [{}]
    assert not any(c.get("trust_remote_code") for c in calls)


def test_load_hf_retries_trust_remote_code_only_when_required(monkeypatch):
    tmm = _load_collect_module()
    calls = []
    rows = [{"text": f"ok{i}", "label": i % 2} for i in range(90)]
    fake = types.ModuleType("datasets")

    def load_dataset(*_args, **kwargs):
        calls.append(dict(kwargs))
        if not kwargs.get("trust_remote_code"):
            raise ValueError("This dataset requires trust_remote_code=True")
        return _FakeTable(rows)

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    texts, labels, family = tmm.load_hf_table(_toy_spec(), limit=80)
    assert family == "single"
    assert len(texts) == 80
    assert len(labels) == 80
    assert calls[0] == {}
    assert calls[1].get("trust_remote_code") is True


def test_load_hf_collects_limit_valid_rows_after_filtering(monkeypatch):
    tmm = _load_collect_module()
    noisy = [{"text": "", "label": 0}] * 120
    valid = [{"text": f"ok{i}", "label": i % 2} for i in range(100)]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *_a, **_k: _FakeTable(noisy + valid)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    texts, labels, _family = tmm.load_hf_table(_toy_spec(), limit=80)
    assert len(texts) == 80
    assert all(t.startswith("ok") for t in texts)
    assert len(labels) == 80

import numpy as np
import pytest

from dataset_complexity_profiler import DatasetProfiler


def _sts_related(n=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, dim))
    scores = np.linspace(0.0, 1.0, n)
    v = u * scores[:, None] + rng.normal(scale=0.05, size=u.shape) * (1.0 - scores[:, None])
    return np.hstack([u, v]), scores


def test_default_meta_model_loads():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    assert profiler.is_fitted is True


def test_fit_transform_smoke(monkeypatch):
    def fake_embed_texts(self, texts, embedder_name, batch_size=64, show_progress=True, model=None):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(texts), 384))
        for i, text in enumerate(texts):
            if text == "b":
                X[i] += 4.0
        return X

    monkeypatch.setattr(DatasetProfiler, "embed_texts", fake_embed_texts)
    profiler = DatasetProfiler()
    texts = ["a", "b"] * 16
    labels = [0, 1] * 16
    Xc = profiler.fit_transform(texts, labels, embedder_name="none", batch_size=2, show_progress=False)
    assert Xc.shape[0] == 32
    assert Xc.ndim == 2


def test_fit_transform_rejects_tiny_demo(monkeypatch):
    from dataset_complexity_profiler.defaults import MIN_SAMPLES_FOR_DIM_RECOMMENDATION

    def fake_embed_texts(self, texts, embedder_name, batch_size=64, show_progress=True, model=None):
        return np.ones((len(texts), 384), dtype=np.float64)

    monkeypatch.setattr(DatasetProfiler, "embed_texts", fake_embed_texts)
    profiler = DatasetProfiler()
    with pytest.raises(ValueError, match=str(MIN_SAMPLES_FOR_DIM_RECOMMENDATION)):
        profiler.fit_transform(["a", "b", "c", "d"], [0, 1, 0, 1], embedder_name="none", show_progress=False)


def test_fit_transform_rejects_single_class_without_meta_model(monkeypatch):
    def fake_embed_texts(self, texts, embedder_name, batch_size=64, show_progress=True, model=None):
        return np.ones((len(texts), 384), dtype=np.float64)

    monkeypatch.setattr(DatasetProfiler, "embed_texts", fake_embed_texts)
    profiler = DatasetProfiler(auto_load_meta_model=False)
    with pytest.raises(ValueError, match="at least 2 classes"):
        profiler.fit_transform(["a"] * 32, [0] * 32, embedder_name="none", show_progress=False)


def test_embed_texts_returns_dense_array():
    class _FakeEmbedder:
        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 8), dtype=np.float32)

    profiler = DatasetProfiler(auto_load_meta_model=False)
    out = profiler.embed_texts(
        ["a"] * 8,
        model=_FakeEmbedder(),
        batch_size=3,
        show_progress=False,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 8)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_recommend_adaptation_strategies():
    profiler = DatasetProfiler(auto_load_meta_model=False)
    easy = profiler.recommend_adaptation(
        sample_count=2000,
        class_count=2,
        baseline_quality=0.95,
        original_dim=384,
        recommended_dim=32,
        minority_fraction=0.45,
    )
    assert easy["strategy"] == "linear_head"

    hard = profiler.recommend_adaptation(
        sample_count=80,
        class_count=20,
        baseline_quality=0.4,
        original_dim=384,
        recommended_dim=128,
        minority_fraction=0.05,
    )
    assert hard["strategy"] in {"adapters", "full_finetune"}
    assert hard["tips"]

    sts_easy = profiler.recommend_adaptation(
        sample_count=40,
        class_count=4,
        baseline_quality=0.82,
        original_dim=384,
        recommended_dim=32,
        task_family="sts",
    )
    assert sts_easy["strategy"] == "linear_head"
    assert "Spearman" in sts_easy["rationale"]


def test_sts_analyze_uses_quartile_class_count_not_n_rows():
    profiler = DatasetProfiler(auto_load_meta_model=False)
    X, y = _sts_related(n=40, dim=8, seed=0)
    report = profiler.analyze_and_adapt(X, y, empirical=True, task_family="sts")
    named_count = int(round(float(report["complexity_profile"]["class_count"])))
    assert report["class_count"] == named_count
    assert report["class_count"] <= 4
    assert report["class_count"] != 40
    assert "dim_is_bucket" in report
    assert report["adaptation_recommendation"]["strategy"] in {
        "linear_head",
        "mlp_head",
        "adapters",
        "full_finetune",
    }
    assert "Spearman" in " ".join(report["adaptation_recommendation"]["tips"])


def test_sts_empirical_distance_labels_match_similarity_dim():
    """|Spearman|: distance gold must not collapse empirical search to dim=2."""
    profiler = DatasetProfiler(auto_load_meta_model=False)
    X, y_sim = _sts_related(n=40, dim=8, seed=0)
    report_sim = profiler.analyze_and_adapt(
        X, y_sim, empirical=True, use_meta_prediction=False, task_family="sts"
    )
    report_dist = profiler.analyze_and_adapt(
        X, -y_sim, empirical=True, use_meta_prediction=False, task_family="sts"
    )
    assert report_sim["baseline_quality"] >= 0.0
    assert report_dist["baseline_quality"] >= 0.0
    assert abs(report_sim["baseline_quality"] - report_dist["baseline_quality"]) < 1e-9
    assert report_sim["recommended_embedding_dim"] == report_dist["recommended_embedding_dim"]


def test_analyze_passes_minority_fraction_into_adaptation():
    profiler = DatasetProfiler(auto_load_meta_model=False)
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal(0.0, 0.3, size=(90, 16)),
            rng.normal(5.0, 0.3, size=(10, 16)),
        ]
    )
    y = np.array([0] * 90 + [1] * 10)
    report = profiler.analyze_and_adapt(X, y, empirical=True)
    tips = " ".join(report["adaptation_recommendation"]["tips"])
    assert "imbalance" in tips.lower()


def test_analyze_refuses_when_linear_probe_is_dummy():
    profiler = DatasetProfiler(auto_load_meta_model=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 8))
    y = rng.integers(0, 20, size=80)
    with pytest.raises(ValueError, match="Cannot recommend a PCA dimension"):
        profiler.analyze_and_adapt(X, y)
    with pytest.raises(ValueError, match="Cannot recommend a PCA dimension"):
        profiler.predict_embedding_dim(X, y)


def test_analyze_text_dataset_sample_limit_on_generators(monkeypatch):
    captured = {}

    def fake_embed(self, texts, **_kwargs):
        captured["n"] = len(texts)
        return np.ones((len(texts), 8), dtype=np.float64)

    def fake_analyze(self, X, labels, **_kwargs):
        captured["y"] = len(labels)
        return {"ok": True}

    monkeypatch.setattr(DatasetProfiler, "embed_texts", fake_embed)
    monkeypatch.setattr(DatasetProfiler, "analyze_and_adapt", fake_analyze)
    profiler = DatasetProfiler(auto_load_meta_model=False)
    report = profiler.analyze_text_dataset(
        (f"t{i}" for i in range(80)),
        (i % 2 for i in range(80)),
        dataset_name="gen",
        sample_limit=32,
        embedder_name="none",
    )
    assert report == {"ok": True, "embedder_name": "none", "sample_limit": 32}
    assert captured["n"] == 32
    assert captured["y"] == 32


def test_analyze_text_dataset_rejects_tiny_before_embed(monkeypatch):
    from dataset_complexity_profiler.defaults import MIN_SAMPLES_FOR_DIM_RECOMMENDATION

    profiler = DatasetProfiler(auto_load_meta_model=False)

    def boom(*_a, **_k):
        raise AssertionError("embedder must not be called before the sample-size gate")

    monkeypatch.setattr(DatasetProfiler, "_load_sentence_transformer", boom)
    monkeypatch.setattr(DatasetProfiler, "embed_texts", boom)
    with pytest.raises(ValueError, match=str(MIN_SAMPLES_FOR_DIM_RECOMMENDATION)):
        profiler.analyze_text_dataset(
            ["a", "b"] * 5,
            [0, 1] * 5,
            dataset_name="tiny",
        )


def test_explicit_broken_meta_model_path_raises(tmp_path):
    missing = tmp_path / "no-such.skops"
    try:
        DatasetProfiler(auto_load_meta_model=True, meta_model_path=str(missing))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for an explicit missing artifact")


def test_analyze_and_adapt_forces_empirical_when_both_modes_off():
    """Regression: disabling predict + empirical used to TypeError on int(None)."""
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(0.0, 0.3, size=(30, 12)),
            rng.normal(4.0, 0.3, size=(30, 12)),
        ]
    )
    y = np.array([0] * 30 + [1] * 30)
    profiler = DatasetProfiler(auto_load_meta_model=False)
    report = profiler.analyze_and_adapt(
        X,
        y,
        dataset_name="both_off",
        use_meta_prediction=False,
        compute_empirical=False,
    )
    assert report["method"] == "empirical"
    assert isinstance(report["recommended_embedding_dim"], int)
    assert report["recommended_embedding_dim"] >= 2

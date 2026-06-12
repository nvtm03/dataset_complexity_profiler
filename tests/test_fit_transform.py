import numpy as np


def test_fit_transform_smoke(monkeypatch):
    """Дымовой тест для `fit_transform` — избегает сетевых вызовов патчем `embed_texts`."""
    from dataset_complexity_profiler import DatasetProfiler

    def fake_embed_texts(self, texts, embedder_name, batch_size=64, show_progress=True):
        # return deterministic random-like array for reproducibility
        return np.ones((len(texts), 384), dtype=float)

    monkeypatch.setattr(DatasetProfiler, "embed_texts", fake_embed_texts)

    profiler = DatasetProfiler()
    Xc = profiler.fit_transform(["a", "b", "c"], [0, 1, 0], embedder_name="none", batch_size=2, show_progress=False)
    assert Xc.shape[0] == 3
    assert Xc.ndim == 2
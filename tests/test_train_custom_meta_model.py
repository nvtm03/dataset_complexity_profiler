import numpy as np


def test_train_custom_meta_model(monkeypatch, tmp_path):
    """Проверяет, что `train_custom_meta_model` обучает мета-модель и её можно сохранить."""
    from dataset_complexity_profiler import DatasetProfiler

    # Patch embedding and intrinsic estimation to be deterministic and fast
    monkeypatch.setattr(
        DatasetProfiler,
        "embed_texts",
        lambda self, texts, embedder_name, batch_size=64, show_progress=True: np.ones((len(texts), 384), dtype=float),
    )

    monkeypatch.setattr(
        DatasetProfiler,
        "estimate_intrinsic_dim",
        lambda self, X, y, quality_threshold=0.95: {"recommended_dim": 2},
    )

    profiler = DatasetProfiler()

    datasets = [
        {"texts": ["a"] * 10, "labels": [0] * 5 + [1] * 5},
        {"texts": ["b"] * 12, "labels": [0] * 6 + [1] * 6},
    ]

    profiler.train_custom_meta_model(datasets, quality_threshold=0.9, embedder_name="none", batch_size=8, show_progress=False, cv=2)
    assert profiler.is_fitted is True

    out_file = tmp_path / "meta_saved.pkl"
    profiler.save_meta_model(str(out_file))
    assert out_file.exists()

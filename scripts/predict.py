"""
Демонстрация работы обученной мета-модели для предсказания оптимальной размерности.
"""

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from dataset_complexity_profiler import DatasetProfiler


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Загрузить переменные окружения из файла `.env`, если HF_TOKEN не задан."""
    if os.environ.get("HF_TOKEN"):
        return
    path = Path(dotenv_path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def predict_optimal_dimension(dataset_id: str, split: str = "train", 
                              sample_limit: int = 300) -> None:
    """Загрузить датасет и предсказать оптимальную размерность с использованием обученной мета-модели."""
    
    load_dotenv()
    
    print(f"Loading dataset {dataset_id}...")
    token = os.environ.get("HF_TOKEN")
    kwargs = {"split": split}
    if token:
        kwargs["token"] = token
    dataset = load_dataset(dataset_id, **kwargs)
    
    if sample_limit and len(dataset) > sample_limit:
        dataset = dataset.shuffle(seed=42).select(range(sample_limit))
    
    texts = dataset["text"]
    labels = dataset["label"]
    
    print(f"Loaded {len(texts)} samples. Encoding with SentenceTransformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    profiler = DatasetProfiler(auto_load_meta_model=False)
    profiler.load_default_meta_model()
    predicted_dim = profiler.predict_embedding_dim(X, labels)
    architecture = profiler.recommend_architecture(
        X.shape[1], predicted_dim, n_classes=len(set(labels))
    )

    print("\n" + "="*70)
    print("META-MODEL PREDICTION")
    print("="*70)
    print(f"Dataset: {dataset_id}")
    print(f"Samples: {len(labels)}")
    print(f"Classes: {len(set(labels))}")
    print(f"Original embedding dimension: {X.shape[1]}")
    print(f"Predicted embedding dimension: {predicted_dim}")
    print(f"Predicted compression ratio: {architecture['compression_ratio']}%")

    output_file = Path("prediction_result.json")
    result = {
        "dataset": dataset_id,
        "sample_count": len(labels),
        "original_dim": int(X.shape[1]),
        "class_count": len(set(labels)),
        "predicted_embedding_dim": int(predicted_dim),
        "architecture_recommendation": architecture,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Prediction saved to {output_file}")

if __name__ == "__main__":
    print("Testing meta-model prediction on new dataset...")
    print("Using: 20 Newsgroups\n")
    
    try:
        predict_optimal_dimension("SetFit/20newsgroups", split="train", sample_limit=300)
    except Exception as e:
        print(f"Note: 20 Newsgroups failed: {e}")
        print("\nFalling back to test on existing benchmark dataset...")
        predict_optimal_dimension("fancyzhx/yelp_polarity", split="train", sample_limit=300)

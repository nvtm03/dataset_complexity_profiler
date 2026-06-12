"""
Скрипт сбора benchmark-данных для обучения мета-модели.
Загружает несколько текстовых датасетов, генерирует эмбеддинги и сохраняет мета-фичи.
"""

import os
import csv
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


def collect_benchmarks(output_csv: str = "benchmarks.csv") -> None:
    """Собрать мета-фичи PyMFE и целевые размерности для текстовых датасетов.

    Параметры:
    - output_csv: имя выходного CSV-файла для результата сбора данных
    """

    load_dotenv()

    benchmark_configs = [
        ("stanfordnlp/imdb", "train", 600, "IMDb (Binary)"),
        ("fancyzhx/ag_news", "train", 600, "AG News (4 classes)"),
        ("dair-ai/emotion", "train", 500, "Emotion (6 classes)"),
        ("cornell-movie-review-data/rotten_tomatoes", "train", 500, "Rotten Tomatoes (Binary)"),
        ("fancyzhx/yelp_polarity", "train", 600, "Yelp Polarity (Binary)"),
    ]

    profiler = DatasetProfiler()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    csv_path = Path(output_csv)
    header_written = False
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for dataset_id, split, sample_limit, display_name in benchmark_configs:
            try:
                print(f"\n=== Processing {display_name} ===")

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
                X = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

                print("Extracting real PyMFE meta-features...")
                feature_names, feature_values = profiler.extract_meta_features(
                    X, labels, return_feature_names=True
                )

                print("Estimating target embedding dimension via search...")
                report = profiler.estimate_intrinsic_dim(
                    X,
                    labels,
                    quality_threshold=0.95,
                )

                sample_count = int(X.shape[0])
                original_dim = int(X.shape[1])
                class_count = int(len(set(labels)))

                if not header_written:
                    header = [
                        "dataset_name",
                        "sample_count",
                        "original_dim",
                        "class_count",
                        "baseline_quality",
                        "intrinsic_dim_estimate",
                        "recommended_dim",
                    ] + feature_names
                    writer.writerow(header)
                    header_written = True

                writer.writerow([
                    display_name,
                    sample_count,
                    original_dim,
                    class_count,
                    float(report["baseline_quality"]),
                    int(report["intrinsic_dim_estimate"]),
                    int(report["recommended_dim"]),
                    *[float(x) for x in feature_values.tolist()],
                ])
                f.flush()

                print(f"✓ Saved: {display_name}")
                print(f"  Recommended dim: {report['recommended_dim']} (from {report['original_dim']})")

            except Exception as e:
                print(f"✗ Failed to process {display_name}: {e}")
                continue

    print(f"\n✓ Benchmark data saved to {output_csv}")


if __name__ == "__main__":
    collect_benchmarks()

"""
Скрипт обучения мета-модели на собранных benchmark-данных.
Использует метаинформацию о датасетах для предсказания оптимальной размерности эмбеддингов.
"""

import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def train_meta_model(
    benchmark_csv: str = "benchmarks.csv",
    output_model: str = "meta_model.pkl",
) -> None:
    """Обучить мета-модель на benchmark-данных."""
    
    csv_path = Path(benchmark_csv)
    if not csv_path.is_file():
        print(f"Error: {benchmark_csv} not found. Run collect_benchmarks.py first.")
        return
    
    print(f"Loading benchmark data from {benchmark_csv}...")
    df = pd.read_csv(csv_path)

    if df.shape[0] < 2:
        print("Error: Need at least 2 benchmark datasets to train the model.")
        return

    print(f"Loaded {df.shape[0]} benchmark records.")

    if "recommended_dim" in df.columns:
        target_column = "recommended_dim"
    elif "recommended_embedding_dim" in df.columns:
        target_column = "recommended_embedding_dim"
    else:
        raise ValueError(
            "Expected 'recommended_dim' or 'recommended_embedding_dim' column in benchmark CSV"
        )

    feature_columns = [
        col for col in df.columns if col not in ["dataset_name", target_column]
    ]

    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    print("\nTraining meta-model on full PyMFE profiles...")
    meta_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    cv_folds = min(3, X.shape[0])
    scores = cross_val_score(meta_model, X, y, cv=cv_folds, scoring="r2")
    print(f"Cross-validation R² scores: {scores}")
    print(f"Mean R² = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    meta_model.fit(X, y)

    output_path = Path(output_model)
    joblib.dump(meta_model, output_path)
    print(f"\n✓ Meta-model saved to {output_model}")

    feature_importance = meta_model.named_steps["regressor"].feature_importances_
    print("\nFeature importance:")
    for name, importance in zip(feature_columns, feature_importance):
        print(f"  {name}: {importance:.6f}")

if __name__ == "__main__":
    train_meta_model()

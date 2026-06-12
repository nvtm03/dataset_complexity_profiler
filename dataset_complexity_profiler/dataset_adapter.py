import warnings
import signal
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_DEFAULT_TIMEOUT_SECONDS = 20


class TimeoutException(Exception):
    """Исключение, которое выбрасывается, если извлечение мета-признаков слишком долго."""
    pass


def _timeout_handler(signum, frame):
    """Обработчик таймаута для извлечения мета-признаков."""
    raise TimeoutException("Meta-feature extraction timed out")


class DatasetProfiler:
    """Инструмент для анализа текстовых датасетов и рекомендации оптимального размера эмбеддингов.

    Основные возможности:
    - Быстро проанализировать датасет и получить рекомендацию по размерности
    - Использовать встроенную мета-модель или обучить свою на ваших данных
    - Сжимать эмбеддинги с помощью PCA при необходимости
    - Получать рекомендации по архитектуре нейросети

    Примеры использования:
    1. Быстрый старт с встроенной моделью: `profiler = DatasetProfiler()`
    2. Анализ датасета: `report = profiler.analyze_text_dataset(texts, labels, "MyDataset")`
    3. Полный конвейер: `X_compressed = profiler.fit_transform(texts, labels)`
    4. Дообучение на своих данных: `profiler.train_custom_meta_model(my_datasets)`
    """

    def __init__(
        self,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        auto_load_meta_model: bool = True,
        meta_model_path: Optional[str] = None,
    ):
        self.imputer = SimpleImputer(strategy="mean")
        self.meta_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
                ),
            ]
        )
        self.is_fitted = False
        self.timeout_seconds = timeout_seconds

        if auto_load_meta_model:
            try:
                self.load_default_meta_model(meta_model_path)
            except Exception:
                pass

    def _prepare_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для анализа: чистит NaN, преобразует типы.

        Возвращает:
        - Кортеж (очищенная матрица признаков, целые метки)
        """
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_clean = self.imputer.fit_transform(X_arr)

        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        if y_arr.dtype.kind not in "biufc":
            y_arr = pd.factorize(y_arr)[0]

        return X_clean, y_arr

    def extract_meta_features(self, X, y, return_feature_names: bool = False):
        """Извлекает мета-признаки датасета с помощью PyMFE.

        Параметры:
        - X: матрица признаков
        - y: метки классов
        - return_feature_names: если True, также возвращает названия признаков

        Возвращает:
        - Массив мета-признаков или кортеж (названия, признаки)
        """
        X_clean, y_arr = self._prepare_data(X, y)

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.timeout_seconds)

        try:
            mfe = MFE(groups=["general", "statistical", "info-theory", "complexity"]) 
            mfe.fit(X_clean, y_arr)
            feature_names, ft_vals = mfe.extract()
            features = np.nan_to_num(ft_vals, nan=0.0, posinf=0.0, neginf=0.0)
            if features.size == 0:
                raise RuntimeError("PyMFE returned an empty feature vector")
            if return_feature_names:
                return list(feature_names), features
            return features
        except TimeoutException as exc:
            raise RuntimeError("Meta-feature extraction timed out") from exc
        except Exception as exc:
            raise RuntimeError("Meta-feature extraction failed") from exc
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

    def build_meta_feature_vector(self, X, y) -> np.ndarray:
        """Строит полный вектор мета-признаков для предсказания мета-моделью.

        Включает базовые статистики датасета (количество образцов, размерность, количество классов)
        и детальные мета-признаки из PyMFE.
        """
        X_clean, y_arr = self._prepare_data(X, y)
        # If labels contain a single class, `_evaluate_quality` would fail (stratify / solver requirements).
        # In that case fall back to a conservative default baseline quality (0.0).
        class_count = int(len(np.unique(y_arr)))
        if class_count < 2:
            baseline_quality = 0.0
        else:
            baseline_quality = float(self._evaluate_quality(X_clean, y_arr))

        dataset_summary = np.array(
            [
                float(X_clean.shape[0]),
                float(X_clean.shape[1]),
                float(class_count),
                baseline_quality,
                float(self._pca_intrinsic_dim(X_clean, target_variance=0.95)),
            ],
            dtype=float,
        )
        raw_features = self.extract_meta_features(X_clean, y_arr)
        features = np.concatenate([dataset_summary, raw_features])
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_meta_model(self, meta_X: np.ndarray, meta_y: np.ndarray) -> None:
        """Обучает мета-модель на уже подготовленных матрицах признаков и целевых значений.

        Параметры:
        - meta_X: матрица мета-признаков (n_datasets, n_features)
        - meta_y: целевые размерности (n_datasets,)
        """
        self.meta_model.fit(meta_X, meta_y)
        self.is_fitted = True

    def load_meta_model(self, model_path: str) -> None:
        """Загружает сохранённую мета-модель с диска."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Meta-model file not found: {model_path}")
        self.meta_model = joblib.load(path)
        self.is_fitted = True

    def save_meta_model(self, model_path: str) -> None:
        """Сохраняет текущую мета-модель на диск."""
        path = Path(model_path)
        joblib.dump(self.meta_model, path)

    def load_default_meta_model(self, model_path: Optional[str] = None) -> None:
        """Загружает встроенную мета-модель, которая идёт в пакете. Если файл не найден, тихо игнорирует ошибку."""
        if model_path is not None:
            self.load_meta_model(model_path)
            return

        default_path = Path(__file__).resolve().parent / "meta_model.pkl"
        if default_path.is_file():
            self.load_meta_model(str(default_path))

    def train_custom_meta_model(
        self,
        datasets: List[Dict],
        quality_threshold: float = 0.95,
        embedder_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        show_progress: bool = True,
        cv: Optional[int] = None,
    ) -> None:
        """Обучает мета-модель на ваших датасетах для адаптации к конкретному домену.

        Каждый датасет в списке — это словарь с ключами `texts` (список строк) и `labels` (список меток).
        Метод:
          1. Эмбеддит тексты
          2. Оценивает оптимальную размерность для каждого датасета
          3. Извлекает мета-признаки
          4. Обучает модель на этих признаках

        Параметры:
        - datasets: список словарей {"texts": [...], "labels": [...]}
        - quality_threshold: нужное качество классификации (по умолчанию 95%)
        - embedder_name: модель для эмбеддинга (по умолчанию all-MiniLM-L6-v2)
        - batch_size: размер батча при эмбеддинге
        - show_progress: показывать ли прогресс
        - cv: если указано, проверяет качество кросс-валидацией
        """
        feature_list = []
        target_list = []

        total = len(datasets)
        if total == 0:
            raise ValueError("No datasets provided for training")

        for idx, ds in enumerate(datasets, start=1):
            print(f"Processing dataset {idx}/{total} for meta-model training...")
            texts = ds.get("texts")
            labels = ds.get("labels")
            if texts is None or labels is None:
                print("  Skipping dataset: missing 'texts' or 'labels'")
                continue

            # Embed
            print(f"  Embedding {len(texts)} texts...")
            X = self.embed_texts(texts, embedder_name=embedder_name, batch_size=batch_size, show_progress=show_progress)

            # Estimate empirical target dimension. If labels lack class diversity, use PCA-based intrinsic dim.
            print("  Estimating intrinsic / recommended dimension via empirical search...")
            if len(np.unique(labels)) < 2:
                recommended = int(self._pca_intrinsic_dim(X, target_variance=0.95))
                print("  -> Single-class labels detected; using PCA intrinsic-dim fallback")
            else:
                info = self.estimate_intrinsic_dim(X, labels, quality_threshold=quality_threshold)
                recommended = int(info.get("recommended_dim", 2))
                print(f"  -> Recommended dim (empirical): {recommended}")

            # Build meta-feature vector
            print("  Building meta-feature vector...")
            feats = self.build_meta_feature_vector(X, labels)
            if feats.ndim != 1:
                feats = np.ravel(feats)

            feature_list.append(feats)
            target_list.append(float(recommended))

        if len(feature_list) < 2:
            raise ValueError("Need at least two datasets with features to train meta-model")

        X_meta = np.vstack(feature_list)
        y_meta = np.array(target_list, dtype=float)

        print(f"Training meta-model on {X_meta.shape[0]} datasets with {X_meta.shape[1]} features each...")

        # Optionally evaluate via cross-validation
        if cv is None:
            cv_folds = min(3, X_meta.shape[0])
        else:
            cv_folds = min(cv, X_meta.shape[0])

        if cv_folds >= 2:
            try:
                from sklearn.model_selection import cross_val_score

                print(f"  Running cross-validation (cv={cv_folds})...")
                scores = cross_val_score(self.meta_model, X_meta, y_meta, cv=cv_folds, scoring="r2")
                print(f"  Cross-validation R²: {scores}")
            except Exception:
                print("  Cross-validation failed or not available; continuing to fit")

        # Fit the meta-model
        self.meta_model.fit(X_meta, y_meta)
        self.is_fitted = True

        # Print feature importance if available
        try:
            feat_imp = self.meta_model.named_steps["regressor"].feature_importances_
            print("Meta-model trained. Feature importances (first 10):", feat_imp[:10])
        except Exception:
            print("Meta-model trained.")

    def predict_embedding_dim(self, X, y, min_dim: int = 2, max_dim: Optional[int] = None) -> int:
        """Предсказывает оптимальный размер эмбеддинга для датасета с помощью мета-модели.

        Параметры:
        - X: матрица эмбеддингов или признаков
        - y: метки классов
        - min_dim: минимальная допустимая размерность
        - max_dim: максимальная допустимая размерность

        Возвращает:
        - Рекомендуемый размер эмбеддинга
        """
        if not self.is_fitted:
            raise ValueError("Meta model is not fitted yet. Call load_meta_model() first.")

        X_clean, y_arr = self._prepare_data(X, y)
        max_dim = int(max_dim or X_clean.shape[1])
        features = self.build_meta_feature_vector(X_clean, y_arr).reshape(1, -1)
        predicted = int(round(self.meta_model.predict(features)[0]))
        return max(min_dim, min(predicted, max_dim))

    def _build_quality_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dims: List[int],
        baseline_score: float,
        threshold: float,
    ) -> Tuple[List[Dict[str, float]], int, float]:
        """Строит кривую зависимости качества классификации от размера эмбеддинга.

        Тестирует разные размеры PCA и оценивает качество LogisticRegression для каждого.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=None
            )
        estimator = LogisticRegression(max_iter=2000, solver="lbfgs")
        curve = []
        best_dim = dims[0]
        best_score = 0.0

        for dim in dims:
            n_components = min(dim, X_train.shape[0], X_train.shape[1])
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            estimator.fit(X_train_pca, y_train)
            score = estimator.score(X_test_pca, y_test)
            curve.append({"dim": dim, "quality": float(score)})

            if score > best_score:
                best_score = score
                best_dim = dim

        threshold_value = baseline_score * threshold
        recommended_dim = next(
            (entry["dim"] for entry in curve if entry["quality"] >= threshold_value),
            best_dim,
        )
        return curve, recommended_dim, best_score

    def estimate_intrinsic_dim(
        self,
        X,
        y,
        quality_threshold: float = 0.95,
        max_dim: Optional[int] = None,
    ) -> Dict[str, object]:
        """Оценивает внутреннюю размерность датасета и ищет оптимальный размер эмбеддинга.

        Параметры:
        - X: матрица эмбеддингов
        - y: метки классов
        - quality_threshold: целевое качество классификации (0.95 = 95%)
        - max_dim: максимальный размер для поиска

        Возвращает:
        - Dict с рекомендациями, оценками размерности и кривой качества
        """
        X_clean, y_arr = self._prepare_data(X, y)
        original_dim = X_clean.shape[1]
        effective_max_dim = int(min(max_dim or original_dim, original_dim, X_clean.shape[0] - 1))

        baseline_score = self._evaluate_quality(X_clean, y_arr)

        if effective_max_dim <= 5:
            dims = list(range(2, effective_max_dim + 1))
        else:
            dims = np.unique(
                np.concatenate(
                    [
                        np.arange(2, min(25, effective_max_dim) + 1, dtype=int),
                        np.linspace(25, effective_max_dim, num=12, dtype=int),
                    ]
                )
            ).tolist()

        curve, recommended_dim, best_score = self._build_quality_curve(
            X_clean, y_arr, dims, baseline_score, quality_threshold
        )

        explained_dim = self._pca_intrinsic_dim(X_clean, target_variance=0.95)

        return {
            "original_dim": original_dim,
            "intrinsic_dim_estimate": int(explained_dim),
            "recommended_dim": int(recommended_dim),
            "recommended_threshold": float(round(baseline_score * quality_threshold, 4)),
            "baseline_quality": float(round(baseline_score, 4)),
            "best_dim_quality": float(round(best_score, 4)),
            "quality_threshold": float(quality_threshold),
            "quality_curve": curve,
        }

    def _evaluate_quality(self, X: np.ndarray, y: np.ndarray) -> float:
        """Оценивает качество классификации на полном наборе признаков (базовая метрика).

        Возвращает:
        - Точность LogisticRegression на тестовом наборе
        """
        if X.shape[1] == 0:
            return 0.0

        estimator = LogisticRegression(max_iter=2000, solver="lbfgs")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=None
            )
        estimator.fit(X_train, y_train)
        return float(estimator.score(X_test, y_test))

    def _pca_intrinsic_dim(self, X: np.ndarray, target_variance: float = 0.95) -> int:
        """Оценивает внутреннюю размерность датасета через PCA.

        Вычисляет количество главных компонент, нужных для объяснения целевого процента дисперсии.
        """
        max_components = min(X.shape[1], X.shape[0] - 1)
        if max_components < 1:
            return 1
        pca = PCA(n_components=max_components)
        pca.fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        idx = int(np.searchsorted(cumulative, target_variance, side="left") + 1)
        return max(1, min(idx, X.shape[1]))

    def recommend_architecture(
        self,
        original_dim: int,
        recommended_dim: int,
        n_classes: Optional[int] = None,
    ) -> Dict[str, object]:
        """Рекомендует архитектуру нейросети на основе размера эмбеддинга и количества классов."""
        width_ratio = recommended_dim / original_dim if original_dim else 1.0
        if n_classes is None:
            n_classes = 2

        if n_classes <= 2:
            model_type = "Binary classifier with a compact embedding and shallow head"
        elif n_classes <= 10:
            model_type = "Multi-class classifier with moderate embedding size"
        else:
            model_type = "Wide embedding + transformer-style head"

        architecture = {
            "model_type": model_type,
            "embedding_dim": recommended_dim,
            "linear_layer_dim": max(16, int(recommended_dim * 1.5)),
            "compression_ratio": float(round((1 - width_ratio) * 100, 2)),
            "note": (
                "Если задача сложнее, сохраняйте пространство не слишком узким. "
                "Рекомендуется сначала протестировать PCA/linear reduction на этой размерности."
            ),
        }
        return architecture

    def analyze_and_adapt(
        self,
        X,
        y,
        dataset_name: str = "Unknown",
        quality_threshold: float = 0.95,
    ) -> Dict[str, object]:
        """Анализирует уже готовые эмбеддинги и выдаёт рекомендации по размерности.

        Параметры:
        - X: матрица эмбеддингов (n_samples, embedding_dim)
        - y: метки классов
        - dataset_name: название датасета для отчёта
        - quality_threshold: требуемое качество классификации

        Возвращает:
        - Dict с рекомендациями, метриками и кривой качества
        """
        X_clean, y_arr = self._prepare_data(X, y)
        meta_features = self.extract_meta_features(X_clean, y_arr)
        dimension_info = self.estimate_intrinsic_dim(
            X_clean, y_arr, quality_threshold=quality_threshold
        )
        arch = self.recommend_architecture(
            dimension_info["original_dim"],
            dimension_info["recommended_dim"],
            n_classes=int(np.unique(y_arr).size),
        )
        prediction = None
        if self.is_fitted:
            prediction = self.predict_embedding_dim(X_clean, y_arr)

        return {
            "dataset_name": dataset_name,
            "sample_count": int(X_clean.shape[0]),
            "original_dim": int(X_clean.shape[1]),
            "class_count": int(np.unique(y_arr).size),
            "baseline_quality": dimension_info["baseline_quality"],
            "intrinsic_dim_estimate": dimension_info["intrinsic_dim_estimate"],
            "recommended_embedding_dim": dimension_info["recommended_dim"],
            "quality_threshold": dimension_info["quality_threshold"],
            "recommended_quality_target": dimension_info["recommended_threshold"],
            "architecture_recommendation": arch,
            "meta_features": meta_features.tolist(),
            "quality_curve": dimension_info["quality_curve"],
            "meta_model_prediction": int(prediction) if prediction is not None else None,
        }

    def embed_texts(
        self,
        texts,
        embedder_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Преобразует список текстов в векторы эмбеддингов.

        Параметры:
        - texts: список строк
        - embedder_name: модель SentenceTransformer для эмбеддинга
        - batch_size: размер батча для обработки
        - show_progress: показывать ли прогресс-бар

        Возвращает:
        - Матрица эмбеддингов (n_samples, embedding_dim)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for text embedding support. "
                "Install it with uv run python -m pip install sentence-transformers"
            ) from exc

        embedder = SentenceTransformer(embedder_name)
        return np.array(
            embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        )

    def fit_transform(
        self,
        texts: List[str],
        labels,
        embedder_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        show_progress: bool = True,
        return_pca: bool = False,
    ) -> np.ndarray:
        """Полный конвейер от текстов к сжатым эмбеддингам. Используйте эту функцию для быстрого старта!

        Шаги:
        1) Эмбеддит тексты
        2) Предсказывает оптимальную размерность с помощью мета-модели
        3) Применяет PCA для сжатия
        4) Возвращает готовые векторы размера (n_samples, optimal_dim)

        Параметры:
        - texts: список текстов
        - labels: метки классов (используются для предсказания размерности)
        - embedder_name: модель для эмбеддинга
        - batch_size: размер батча
        - show_progress: показывать ли прогресс
        - return_pca: если True, возвращает также объект PCA

        Возвращает:
        - Сжатые эмбеддинги формы (n_samples, optimal_dim)
        - Если return_pca=True, возвращает кортеж (X_compressed, pca)
        """
        print(f"Step 1/5: Embedding {len(texts)} texts using '{embedder_name}'...")
        X_raw = self.embed_texts(
            texts, embedder_name=embedder_name, batch_size=batch_size, show_progress=show_progress
        )
        print(f"  -> Embedded into shape: {X_raw.shape}")

        # Determine optimal dimension
        if self.is_fitted:
            print("Step 2/5: Meta-model loaded — predicting optimal dimension using meta-model...")
            optimal_dim = int(self.predict_embedding_dim(X_raw, labels))
        else:
            print("Step 2/5: Meta-model not loaded — estimating intrinsic dimension via PCA variance...")
            optimal_dim = int(self._pca_intrinsic_dim(X_raw, target_variance=0.95))

        optimal_dim = max(1, int(optimal_dim))
        optimal_dim = min(optimal_dim, X_raw.shape[0], X_raw.shape[1])
        print(f"  -> Chosen optimal dimension: {optimal_dim}")

        # PCA compression
        print("Step 3/5: Initializing PCA...")
        pca = PCA(n_components=optimal_dim)

        print("Step 4/5: Fitting PCA and transforming embeddings...")
        X_compressed = pca.fit_transform(X_raw)

        print("Step 5/5: Compression complete — returning compressed embeddings.")
        X_arr = np.asarray(X_compressed)
        if return_pca:
            return X_arr, pca
        return X_arr

    def analyze_text_dataset(
        self,
        texts,
        labels,
        dataset_name: str,
        embedder_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        sample_limit: Optional[int] = None,
        quality_threshold: float = 0.95,
    ) -> Dict[str, object]:
        """Анализирует датасет от начала до конца: эмбеддит тексты и выдаёт полный отчёт.

        Параметры:
        - texts: список текстов для анализа
        - labels: метки классов
        - dataset_name: название датасета (для отчёта)
        - embedder_name: модель для эмбеддинга
        - batch_size: размер батча
        - sample_limit: если указано, анализирует только первые N образцов
        - quality_threshold: требуемое качество классификации

        Возвращает:
        - Полный отчёт с рекомендациями, метриками и кривой качества
        """
        if sample_limit is not None:
            texts = texts[:sample_limit]
            labels = labels[:sample_limit]

        X = self.embed_texts(
            texts,
            embedder_name=embedder_name,
            batch_size=batch_size,
            show_progress=True,
        )
        report = self.analyze_and_adapt(
            X,
            labels,
            dataset_name=dataset_name,
            quality_threshold=quality_threshold,
        )
        report["embedder_name"] = embedder_name
        report["sample_limit"] = int(sample_limit) if sample_limit is not None else None
        return report

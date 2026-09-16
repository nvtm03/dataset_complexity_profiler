"""Facade: embed texts, extract meta-features, and recommend a PCA dimension."""

from __future__ import annotations

import logging
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA

from .complexity_report import interpret_id_profile
from .defaults import (
    DEFAULT_EMBEDDER_NAME,
    DEFAULT_QUALITY_THRESHOLD,
    check_min_samples_for_recommendation,
    dim_is_bucket,
    feasible_dim_bucket,
    normalize_task_family,
)
from .empirical import (
    estimate_intrinsic_dim as _estimate_intrinsic_dim,
)
from .empirical import (
    pca_intrinsic_dim,
)
from .features import (
    _build_meta_feature_vector_prepared,
    _coerce_label_series,
    id_profile_from_named,
    prepare_data,
)
from .features import (
    build_meta_feature_vector as _build_meta_feature_vector,
)
from .predict import (
    is_recoverable_predict_error,
    load_default_meta_artifact,
    load_meta_artifact,
)
from .predict import (
    predict_embedding_dim as _predict_embedding_dim,
)
from .probes import (
    linear_separability_status,
    pair_features,
    require_linear_separability,
)

logger = logging.getLogger(__name__)


def _prefix(seq: Any, n: Optional[int]) -> Any:
    if n is None:
        return seq
    try:
        return seq[:n]
    except TypeError:
        return list(islice(seq, n))


class DatasetProfiler:
    """Recommend a compact PCA dim via packaged RF or an empirical probe grid."""

    def __init__(
        self,
        auto_load_meta_model: bool = True,
        meta_model_path: Optional[str] = None,
    ):
        self.meta_model: Any = None
        self.is_fitted = False
        self._meta_feature_names: List[str] = []

        if auto_load_meta_model:
            try:
                self.load_default_meta_model(meta_model_path)
            except (FileNotFoundError, OSError, ValueError) as orig:
                if meta_model_path is not None:
                    raise
                logger.warning("Packaged meta-model not loaded: %s", orig)
                self.is_fitted = False

    def build_meta_feature_vector(
        self,
        X: Any,
        y: Any,
        task_family: str = "single",
    ) -> Dict[str, float]:
        return _build_meta_feature_vector(X, y, task_family=task_family)

    def load_meta_model(self, model_path: str) -> None:
        self.meta_model, self._meta_feature_names = load_meta_artifact(model_path)
        self.is_fitted = True

    def load_default_meta_model(self, model_path: Optional[str] = None) -> None:
        """Load the packaged artifact or ``model_path``. Missing file leaves ``is_fitted`` false."""
        loaded = load_default_meta_artifact(model_path)
        if loaded is None:
            return
        self.meta_model, self._meta_feature_names = loaded
        self.is_fitted = True

    def predict_embedding_dim(
        self,
        X: Any,
        y: Any,
        task_family: str = "single",
        features: Optional[Union[Dict[str, float], np.ndarray]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> int:
        """Return a PCA-dim bucket, clipped to the matrix rank."""
        if not self.is_fitted:
            raise ValueError("Meta model is not fitted yet. Call load_meta_model() first.")
        return _predict_embedding_dim(
            X,
            y,
            self.meta_model,
            meta_feature_names=self._meta_feature_names,
            task_family=task_family,
            features=features,
            feature_names=feature_names,
        )

    def estimate_intrinsic_dim(
        self,
        X: Any,
        y: Any,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        max_dim: Optional[int] = None,
        min_dim_bound: int = 2,
        task_family: str = "single",
        stop_at_threshold: bool = False,
        precomputed_baseline: Optional[float] = None,
        precomputed_pca95: Optional[float] = None,
    ) -> Dict[str, object]:
        return _estimate_intrinsic_dim(
            X,
            y,
            quality_threshold=quality_threshold,
            max_dim=max_dim,
            min_dim_bound=min_dim_bound,
            task_family=task_family,
            stop_at_threshold=stop_at_threshold,
            precomputed_baseline=precomputed_baseline,
            precomputed_pca95=precomputed_pca95,
        )

    def recommend_adaptation(
        self,
        sample_count: int,
        class_count: int,
        baseline_quality: float,
        original_dim: int,
        recommended_dim: int,
        minority_fraction: Optional[float] = None,
        task_family: str = "single",
    ) -> Dict[str, object]:
        """Pick an adaptation strategy with deterministic rules (no LLM, no model)."""
        samples_per_class = sample_count / max(class_count, 1)
        compression_ratio = 1.0 - (recommended_dim / original_dim) if original_dim else 0.0
        minority = 1.0 if minority_fraction is None else float(minority_fraction)
        strategy_rank = {
            "linear_head": 1,
            "mlp_head": 2,
            "adapters": 3,
            "full_finetune": 4,
        }
        head_by_strategy = {
            "linear_head": "Ridge/Logistic",
            "mlp_head": "MLP(1-2 layers)",
            "adapters": "Encoder adapters + linear head",
            "full_finetune": "Full encoder fine-tune + classification head",
        }

        if str(task_family) == "sts":
            rho = abs(float(baseline_quality))
            if rho >= 0.70:
                base_strategy = "linear_head"
                rationale = (
                    "Cosine ranking on frozen embeddings already matches gold scores "
                    "(|Spearman| ≥ 0.70) — a linear similarity head is enough."
                )
            elif rho >= 0.40:
                base_strategy = "mlp_head"
                rationale = (
                    "Moderate |Spearman|; a small MLP on frozen embeddings is a better "
                    "fit than a linear cosine head."
                )
            elif rho >= 0.20:
                base_strategy = "adapters"
                rationale = (
                    "Weak ranking signal; parameter-efficient fine-tuning of the encoder "
                    "(LoRA/adapters) is the next step."
                )
            else:
                base_strategy = "full_finetune"
                rationale = (
                    "Near-chance |Spearman|; the encoder needs a full fine-tune with "
                    "strong regularization."
                )
        elif baseline_quality >= 0.90 and samples_per_class >= 50:
            base_strategy = "linear_head"
            rationale = (
                "A linear classifier on frozen embeddings already scores high — "
                "tuning the head is enough."
            )
        elif baseline_quality >= 0.75 and samples_per_class >= 30:
            base_strategy = "mlp_head"
            rationale = (
                "Linear-head quality is moderate; a 1–2 layer MLP on the embeddings "
                "with light regularization is a better fit."
            )
        elif samples_per_class >= 20:
            base_strategy = "adapters"
            rationale = (
                "The task is harder than linear separation; use parameter-efficient "
                "encoder fine-tuning (LoRA/adapters)."
            )
        else:
            base_strategy = "full_finetune"
            rationale = (
                "Few examples per class and/or a weak baseline — full encoder fine-tune "
                "with strong regularization and augmentation."
            )

        tips: List[str] = [
            f"Recommended PCA embedding dim: {recommended_dim} "
            f"(~{compression_ratio * 100:.1f}% compression).",
        ]
        if str(task_family) == "sts":
            tips.append(f"|Spearman| vs gold ≈ {abs(float(baseline_quality)):.2f}.")
        else:
            tips.append(f"Examples per class ≈ {samples_per_class:.1f}.")
        if str(task_family) != "sts" and minority < 0.15:
            tips.append(
                "Strong class imbalance: use class weights / focal loss / resampling."
            )
        if str(task_family) != "sts" and class_count > 20:
            tips.append(
                "Many classes: increase head capacity and watch rare-class recall."
            )
        if compression_ratio > 0.7:
            tips.append(
                "Heavy compression is fine at the current baseline; run the empirical "
                "PCA grid if the bucket looks too small for a hard domain."
            )

        return {
            "strategy": base_strategy,
            "strategy_rank": strategy_rank[base_strategy],
            "base_strategy": base_strategy,
            "rationale": rationale,
            "tips": tips,
            "samples_per_class": round(samples_per_class, 2),
            "suggested_embedding_dim": int(recommended_dim),
            "suggested_head": head_by_strategy[base_strategy],
        }

    def analyze_and_adapt(
        self,
        X: Any,
        y: Any,
        dataset_name: str = "Unknown",
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        min_dim_bound: int = 2,
        use_meta_prediction: bool = True,
        empirical: bool = False,
        compute_empirical: Optional[bool] = None,
        task_family: str = "single",
    ) -> Dict[str, Any]:
        """Build features once; run empirical search if requested or if predict is unavailable.

        ``compute_empirical`` is a deprecated alias of ``empirical``.
        """
        if compute_empirical is not None:
            empirical = bool(compute_empirical)

        task_family = normalize_task_family(task_family)
        X_clean, y_arr = prepare_data(X, y, task_family=task_family)
        check_min_samples_for_recommendation(X_clean.shape[0])
        named = _build_meta_feature_vector_prepared(X_clean, y_arr, task_family=task_family)
        feature_names = list(named.keys())
        meta_features = [float(named[name]) for name in feature_names]
        require_linear_separability(
            linear_separability_status(
                float(named.get("baseline_quality") or 0.0),
                y_arr,
                task_family=task_family,
            ),
            dataset_name=dataset_name,
        )

        run_predict = bool(use_meta_prediction) and self.is_fitted and not empirical
        prediction = None
        if run_predict:
            try:
                prediction = self.predict_embedding_dim(
                    X_clean,
                    y_arr,
                    task_family=task_family,
                    features=named,
                    feature_names=feature_names,
                )
            except ValueError as exc:
                if not is_recoverable_predict_error(exc):
                    raise
                prediction = None

        if prediction is None and not empirical:
            logger.warning(
                "%s: no meta-model prediction available, forcing the empirical PCA grid",
                dataset_name,
            )
        run_empirical = empirical or prediction is None
        if run_empirical:
            dimension_info = _estimate_intrinsic_dim(
                X_clean,
                y_arr,
                quality_threshold=quality_threshold,
                min_dim_bound=min_dim_bound,
                task_family=task_family,
                precomputed_baseline=float(named.get("baseline_quality") or 0.0),
                precomputed_pca95=float(named.get("intrinsic_dim_pca_95") or 0.0),
            )
            empirical_dim = dimension_info["recommended_dim"]
            method = "empirical"
        else:
            original_dim = X_clean.shape[1]
            baseline = float(named.get("baseline_quality") or 0.0)
            pca95 = int(round(float(named.get("intrinsic_dim_pca_95") or 0.0)))
            dimension_info = {
                "original_dim": original_dim,
                "intrinsic_dim_estimate": pca95,
                "recommended_dim": int(prediction),
                "recommended_threshold": round(baseline * quality_threshold, 4),
                "baseline_quality": baseline,
                "best_dim_quality": None,
                "quality_threshold": float(quality_threshold),
                "quality_curve": [],
                "min_dim_bound": int(min_dim_bound),
                "task_family": task_family,
            }
            empirical_dim = None
            method = "predictive"

        linear_sep = linear_separability_status(
            float(dimension_info["baseline_quality"]),
            y_arr,
            task_family=task_family,
        )
        dimension_info["linear_separability"] = linear_sep

        if prediction is not None:
            chosen_dim = int(prediction)
        elif empirical_dim is not None:
            chosen_dim = int(empirical_dim)
        else:
            raise RuntimeError(
                f"{dataset_name}: cannot recommend a dimension without a meta-model "
                "prediction or an empirical PCA search"
            )
        original_dim = dimension_info["original_dim"]
        ratio_den = int(original_dim)
        if str(task_family) == "sts" and ratio_den >= 2:
            ratio_den = ratio_den // 2
        compression_ratio_pct = (
            round((1 - chosen_dim / ratio_den) * 100, 2) if ratio_den else 0.0
        )
        id_profile = id_profile_from_named(named)
        id_interpretation = interpret_id_profile(id_profile)
        class_count = int(round(float(named.get("class_count") or 0.0)))
        minority_fraction: Optional[float] = None
        if str(task_family) != "sts" and y_arr.size:
            _, counts = np.unique(y_arr, return_counts=True)
            minority_fraction = float(np.min(counts) / y_arr.size)
        adaptation = self.recommend_adaptation(
            sample_count=X_clean.shape[0],
            class_count=class_count,
            baseline_quality=float(dimension_info["baseline_quality"]),
            original_dim=ratio_den,
            recommended_dim=chosen_dim,
            minority_fraction=minority_fraction,
            task_family=task_family,
        )

        return {
            "dataset_name": dataset_name,
            "task_family": task_family,
            "method": method,
            "sample_count": X_clean.shape[0],
            "original_dim": X_clean.shape[1],
            "class_count": class_count,
            "baseline_quality": dimension_info["baseline_quality"],
            "intrinsic_dim_estimate": dimension_info["intrinsic_dim_estimate"],
            "recommended_embedding_dim": chosen_dim,
            "empirical_recommended_dim": empirical_dim,
            "quality_threshold": dimension_info["quality_threshold"],
            "recommended_quality_target": dimension_info["recommended_threshold"],
            "compression_ratio_pct": compression_ratio_pct,
            "complexity_profile": dict(named),
            "id_profile": id_profile,
            "id_interpretation": id_interpretation,
            "adaptation_recommendation": adaptation,
            "meta_feature_names": feature_names,
            "meta_features": meta_features,
            "quality_curve": dimension_info["quality_curve"],
            "meta_model_prediction": int(prediction) if prediction is not None else None,
            "linear_separability": linear_sep,
            "min_dim_bound": int(min_dim_bound),
            "dim_is_bucket": dim_is_bucket(chosen_dim),
        }

    def _load_sentence_transformer(self, embedder_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is an optional extra used for text embedding. "
                "Install it with: pip install 'dataset-complexity-profiler[text]'"
            ) from exc
        return SentenceTransformer(embedder_name)

    def embed_texts(
        self,
        texts: Sequence[str],
        embedder_name: str = DEFAULT_EMBEDDER_NAME,
        batch_size: int = 64,
        show_progress: bool = True,
        model: Any = None,
    ) -> np.ndarray:
        """Encode raw strings with SentenceTransformer. Requires the ``[text]`` extra."""
        texts = list(texts)
        if any(not isinstance(t, str) for t in texts):
            raise ValueError("All texts must be strings.")
        embedder = model if model is not None else self._load_sentence_transformer(embedder_name)
        return np.asarray(
            embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        )

    def fit_transform(
        self,
        texts: Sequence[str],
        labels: Sequence[Any],
        embedder_name: str = DEFAULT_EMBEDDER_NAME,
        batch_size: int = 64,
        show_progress: bool = True,
        return_pca: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, PCA]]:
        """Embed texts, predict a feasible PCA dim, and return the compressed matrix."""
        if len(texts) != len(labels):
            raise ValueError(
                f"X and y length mismatch: {len(texts)} vs {len(labels)}"
            )
        check_min_samples_for_recommendation(len(texts))
        if len(_coerce_label_series(labels).dropna().unique()) < 2:
            raise ValueError("Dataset must contain at least 2 classes.")
        logger.info("Embedding %s texts with '%s'", len(texts), embedder_name)
        X_raw = self.embed_texts(
            texts, embedder_name=embedder_name, batch_size=batch_size, show_progress=show_progress
        )
        X_clean, y_clean = prepare_data(X_raw, labels)
        logger.info("Embedded shape: %s", X_clean.shape)

        if self.is_fitted:
            logger.info("Predicting dimension with the packaged meta-model")
            try:
                optimal_dim = int(self.predict_embedding_dim(X_clean, y_clean))
            except ValueError as exc:
                if not is_recoverable_predict_error(exc):
                    raise
                logger.warning("Meta-model predict failed (%s); falling back to PCA-95", exc)
                optimal_dim = pca_intrinsic_dim(X_clean, target_variance=0.95)
        else:
            logger.info("Meta-model not loaded; using PCA-95 variance dim")
            optimal_dim = pca_intrinsic_dim(X_clean, target_variance=0.95)

        optimal_dim = feasible_dim_bucket(optimal_dim, X_clean.shape[1], X_clean.shape[0])
        logger.info("Chosen dimension: %s", optimal_dim)

        pca = PCA(n_components=optimal_dim)
        X_compressed = pca.fit_transform(X_clean)
        X_arr = np.asarray(X_compressed)
        if return_pca:
            return X_arr, pca
        return X_arr

    def analyze_text_dataset(
        self,
        texts: Sequence[str],
        labels: Sequence[Any],
        dataset_name: str,
        embedder_name: str = DEFAULT_EMBEDDER_NAME,
        batch_size: int = 64,
        sample_limit: Optional[int] = None,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        task_family: str = "single",
        texts_b: Optional[Sequence[str]] = None,
        empirical: bool = False,
    ) -> Dict[str, Any]:
        """Embed a text dataset (single / pair / STS) and run :meth:`analyze_and_adapt`."""
        if sample_limit is not None and sample_limit < 1:
            raise ValueError(f"sample_limit must be >= 1, got {sample_limit}")

        texts = list(_prefix(texts, sample_limit))
        labels_series = _coerce_label_series(_prefix(labels, sample_limit))
        if texts_b is not None:
            texts_b = list(_prefix(texts_b, sample_limit))

        labels = labels_series.tolist()
        n = len(texts)
        if n != len(labels):
            raise ValueError(f"X and y length mismatch: {n} vs {len(labels)}")

        family = normalize_task_family(task_family)
        if family in {"pair", "sts"}:
            if texts_b is None:
                raise ValueError("task_family pair/sts requires texts_b")
            if len(texts_b) != n:
                raise ValueError(
                    f"X and y length mismatch: texts_b has {len(texts_b)} rows, texts has {n}"
                )
        check_min_samples_for_recommendation(n)
        if family != "sts":
            n_classes = len(labels_series.dropna().unique())
            if n_classes < 2:
                raise ValueError("Dataset must contain at least 2 classes.")

        if family in {"pair", "sts"}:
            embedder = self._load_sentence_transformer(embedder_name)
            encode_kw = dict(
                embedder_name=embedder_name,
                batch_size=batch_size,
                show_progress=True,
                model=embedder,
            )
            u = self.embed_texts(texts, **encode_kw)
            v = self.embed_texts(texts_b, **encode_kw)
            X = pair_features(u, v) if family == "pair" else np.hstack([u, v])
        else:
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
            task_family=family,
            empirical=empirical,
        )
        report["embedder_name"] = embedder_name
        report["sample_limit"] = int(sample_limit) if sample_limit is not None else None
        return report

# Dataset Complexity Profiler

[English](README_en.md) | [Русский](README.md)

[![CI](https://github.com/nvtm03/dataset_complexity_profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/nvtm03/dataset_complexity_profiler/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Package version: **0.3.0** (`pyproject.toml`).

A library that profiles a labeled text dataset in SentenceTransformer embedding space and recommends a compact PCA dimension. On typical tasks a linear probe retains its quality with 16–32 principal components instead of the original 384 coordinates.

Modes:

- **Predictive** (default). Five features are passed to the bundled `RandomForestClassifier`. The model returns a value from the discrete grid `{4, 8, 16, 32, 64, 128, 256}`.
- **Empirical**. PCA is estimated only on the training split of each cross-validation fold. The search returns the smallest dimension at which probe quality is at least 97% of the full-vector score.

Every public entry point, including `estimate_intrinsic_dim`, requires at least 30 rows. `analyze_text_dataset` and the CLI enforce this check before MiniLM is loaded.

If the baseline quality is indistinguishable from random chance (adjusted accuracy gain below 0.10; for STS, |Spearman| below 0.15), dimensionality reduction is not applicable and the tool raises a `ValueError`.

Meta-model quality is reported as exact grid-level Accuracy and Adjacent Accuracy (error of at most one grid step). On a log grid with step ×2, “±1 level” is the window `[d/2, 2d]`. A constant “always 8” baseline scores Adjacent Accuracy 0.65; the model on the combined training set scores Accuracy 0.43, Adjacent Accuracy 0.79. Label agreement across resamples of the same corpus is ≈ 0.60 / 0.87. Details: [`docs/META_MODEL_METRICS.md`](docs/META_MODEL_METRICS.md).

The artifact is serialized with **skops** (scikit-learn **1.6.1**). Runtime requirement: `scikit-learn>=1.6.1`. A minor sklearn mismatch logs a warning and does not block loading.

## Installation

```bash
pip install -e .
pip install -e ".[text]"        # SentenceTransformer and Hugging Face loaders
pip install -e ".[dev,text]"    # tests and wheel build
```

The core package operates on a pre-computed numeric embedding matrix (no PyTorch dependency). To use the built-in Hugging Face encoders for raw text, install the `[text]` extra.

```python
from dataset_complexity_profiler import DatasetProfiler

texts = [
    "This movie is great!",
    "Terrible plot.",
    "I loved the acting.",
    "Worst film ever.",
] * 8
labels = [1, 0, 1, 0] * 8

profiler = DatasetProfiler()
report = profiler.analyze_text_dataset(texts, labels, dataset_name="demo")
print(report["recommended_embedding_dim"])
print(report["adaptation_recommendation"]["strategy"])

X_ready = profiler.fit_transform(texts, labels)
print(X_ready.shape)
```

```bash
dcp analyze --hf fancyzhx/ag_news --limit 400 --output report.json
dcp analyze --csv data.csv --text-col text --label-col label --limit 500
dcp analyze --csv data.csv --empirical
```

`--empirical` runs the full PCA search. `--limit` is optional.

## Bundled artifact

The two CSV files have different roles and are not interchangeable.

- [`research/feature_selection.csv`](research/feature_selection.csv) — feature-selection table (~240×131; ≈ 196 rows / 57 groups after the linear-separability filter). Contains PyMFE columns. The five-feature vs PyMFE comparison runs only on this table (`research/feature_selection.ipynb`).
- [`research/training_extra.csv`](research/training_extra.csv) — training supplement (58 sources, five features and filter columns; no PyMFE). It is excluded from the family comparison: missing PyMFE values would introduce a batch effect.

The bundled Random Forest is trained on the concatenated dataset after applying the numerical linear-separability filter (`linear_sep_value` vs `linear_sep_threshold`): ≈ 254 rows / 115 groups. The supplement takes part in cross-validation and in the final fit. Features: `class_count`, `baseline_quality`, `mean_centroid_cosine_distance`, `intrinsic_dim_twonn`, `intrinsic_dim_pca_95`. `GroupKFold` splits by source (name without the `[variant]` suffix, lower-cased).

The bundled artifact is reproduced only with both CSV files. If `--train-extra-csv` points to a missing file, training aborts. Omitting the flag trains on `--selection-csv` alone (a custom table, not the bundled model).

```bash
python research/train_meta_model.py \
  --selection-csv research/feature_selection.csv \
  --train-extra-csv research/training_extra.csv
```

The script writes `src/dataset_complexity_profiler/meta_model.skops`. PyMFE is not imported at runtime.

Filter: κ_m = (accuracy − majority) / (1 − majority) ≥ 0.10; for STS, |Spearman| ≥ 0.15. Encoder: `paraphrase-multilingual-MiniLM-L12-v2`. Literature: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md). Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Custom model retraining

The inference library is decoupled from the training pipeline. The meta-model can be retrained on domain-specific datasets (for example medical or legal corpora) by preparing a CSV that follows the training feature contract and running:

```bash
python research/train_meta_model.py \
  --selection-csv custom_corporate_data.csv \
  --output-model src/dataset_complexity_profiler/meta_model.skops
```

The table must include the `recommended_dim` target, the five contract features, and the numeric linear-separability columns. With two sources, as in the bundled artifact, pass the second file via `--train-extra-csv`. After the `.skops` file is written, `DatasetProfiler()` loads the new model on the next instantiation.

## Repository layout

```
src/dataset_complexity_profiler/   # library, meta_model.skops, CLI
research/                          # training, collection, CSV, selection notebook
docs/
tests/
```

## License

MIT. See [LICENSE](LICENSE).

# Dataset Complexity Profiler

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Библиотека анализирует размеченный текстовый датасет в пространстве эмбеддингов SentenceTransformer и рекомендует оптимальную размерность для сжатия через PCA. На типовых задачах линейному классификатору достаточно 16–32 главных компонент вместо исходных 384 признаков.

Режимы:

- **Прогностический** (по умолчанию). Пять признаков передаются в поставляемый `RandomForestClassifier`. Модель возвращает значение из дискретной сетки `{4, 8, 16, 32, 64, 128, 256}`.
- **Эмпирический**. PCA обучается строго на train-фолдах кросс-валидации. Выбирается наименьшая размерность, при которой линейная модель сохраняет не менее 97% качества относительно исходного полного вектора.

Ограничение на размер данных: библиотека требует минимум 30 примеров. Проверка выборки срабатывает сразу — ещё до загрузки весов модели в память.

Если в данных нет явного сигнала и базовая модель работает не лучше случайного угадывания (прирост точности < 0.10 или ∣Spearman∣<0.15 для STS), сжимать эмбеддинги бессмысленно — метод выбросит `ValueError`.

**Качество предсказаний:**  
Модель оценивается по двум метрикам: точному попаданию в размерность (**Accuracy**) и ошибке не более чем на один шаг сетки (**Adjacent Accuracy**, то есть попадание в диапазон `[d/2, 2d]`).

- **Точное попадание (Accuracy):** 0.43 (при случайном бейзлайне 0.16).
- **Попадание в соседний бакет (Adjacent Accuracy):** 0.79 (против 0.65 у наивного правила «всегда брать 8»).
  Погрешность в основном связана с шумом разметки в самих текстах, а не со слабостью алгоритма. Подробный разбор и валидация: [`docs/META_MODEL_METRICS.md`](docs/META_MODEL_METRICS.md).

Артефакт сериализован через **skops**. Среда выполнения: `scikit-learn>=1.6.1`. Несовпадение минорной версии sklearn даёт warning, загрузка не блокируется.

## Установка

Предобученная модель уже встроена в пакет и работает сразу из коробки.

### 1. Через uv (рекомендуется)

```bash
# Базовая версия (работает с готовыми матрицами эмбеддингов):
uv add dataset-complexity-profiler

# Полная версия (с поддержкой сырого текста через PyTorch и Hugging Face):
uv add "dataset-complexity-profiler[text]"

# Или запуск без установки в окружение:
uvx --from "dataset-complexity-profiler[text]" dcp analyze --help
```

### 2. Через pip

```bash
# Базовая версия:
pip install dataset-complexity-profiler

# Полная версия:
pip install "dataset-complexity-profiler[text]"
```

### 3. Установка из исходников (для разработки и тестов)

Если вы хотите запустить тесты, переобучить модель или внести правки:

```bash
git clone https://github.com/nvtm03/dataset_complexity_profiler.git
cd dataset_complexity_profiler

# Быстрая синхронизация окружения через uv:
uv sync --extra dev --extra text

# Или классический editable install через pip:
pip install -e ".[dev,text]"
```

## Быстрый старт

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

### Использование через командную строку

Библиотека предоставляет консольную команду `dcp`:

```bash
# Быстрый анализ датасета с Hugging Face с сохранением в JSON:
dcp analyze --hf fancyzhx/ag_news --limit 400 --output report.json

# Анализ локального CSV-файла:
dcp analyze --csv data.csv --text-col text --label-col label --limit 500

# Точный эмпирический подбор размерности полным перебором PCA:
dcp analyze --csv data.csv --empirical
```

`--empirical` запускает полный перебор размерностей PCA. `--limit` необязателен.

## Предобученная мета-модель

В пакет встроен готовый легковесный классификатор (`meta_model.skops`), который сразу доступен после установки и не требует обучения.

- **На чём обучен:** объединенная выборка из ~250 текстовых корпусов (классификация, NLI, STS) с Hugging Face.
- **Входные признаки:** 5 геометрических мета-метрик датасета (число классов, точность бейзлайна, расстояние центроидов, оценки размерности TwoNN и PCA-95).
- **Валидация:** честный `GroupKFold` по источникам датасетов во избежание утечек между похожими выборками.

- **Базовый энкодер:** `paraphrase-multilingual-MiniLM-L12-v2` (384-d).
- **Среда выполнения:** библиотека не тянет PyMFE в runtime (он использовался только на этапе отбора признаков).

Подробное описание обучающих выборок, порогов и метрик качества вынесено в [`docs/META_MODEL_METRICS.md`](docs/META_MODEL_METRICS.md).

## Воспроизводимость и дообучение модели

Код инференса намеренно отделен от пайплайна обучения: саму мета-модель можно воспроизвести с нуля или переобучить под специфичный домен (например, медицинские или юридические тексты).

### 1. Воспроизведение поставляемой модели

Обучение выполняется на объединении бенчмарков из `feature_selection.csv` и `training_extra.csv` (суммарно 254 датасета):

```bash
python research/train_meta_model.py \
  --selection-csv research/feature_selection.csv \
  --train-extra-csv research/training_extra.csv
```

Скрипт запишет обученную модель в `src/dataset_complexity_profiler/meta_model.skops`

### 2. Обучение на собственных данных

Если вам нужно адаптировать профилировщик под узкий домен, подготовьте таблицу мета-признаков (целевая колонка `recommended_dim`, 5 обязательных признаков и порог разделимости) и передайте её в скрипт:

```bash
python research/train_meta_model.py \
  --selection-csv custom_domain_meta_features.csv \
  --output-model src/dataset_complexity_profiler/meta_model.skops
```

## Структура репозитория

```text
src/dataset_complexity_profiler/   # ядро библиотеки, meta_model.skops, CLI
research/                          # скрипты обучения, сбор данных, CSV, ноутбуки
docs/                              # архитектура, метрики и обзор литературы
tests/                             # модульные и интеграционные тесты
```

## Лицензия

[LICENSE](LICENSE)

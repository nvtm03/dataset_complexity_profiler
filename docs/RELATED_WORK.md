# Связанные работы и теоретическая база

`Dataset Complexity Profiler` представляет собой прикладную реализацию классической схемы мета-обучения:
**характеристика задачи → мета-признаки → выбор действия** в пространстве плотных векторных представлений текстов (`SentenceTransformer`).

---

## Связь с моделью выбора алгоритмов Райса (Rice Framework)

Библиотека реализует классическую структуру *Algorithm Selection Problem* (Rice, 1976; Kerschke et al., 2019):

| Компонент модели Райса | Реализация в Dataset Complexity Profiler |
| :--- | :--- |
| **Пространство задач** (*Problem Space*) | Размеченные текстовые датасеты (`single`, `pair`, `sts`) в пространстве эмбеддингов MiniLM (384 измерения). |
| **Пространство мета-признаков** (*Feature Space*) | Вектор из 5 геометрических метрик (`class_count`, `baseline_quality`, `mean_centroid_cosine_distance`, `intrinsic_dim_twonn`, `intrinsic_dim_pca_95`). |
| **Пространство действий** (*Algorithm Space*) | 1) Дискретный бакет размерности PCA ∈ {4, 8, 16, 32, 64, 128, 256}.<br>2) Рекомендация стратегии адаптации энкодера (`linear_head`, `mlp_head`, `adapters`, `full_finetune`). |
| **Критерий качества** (*Performance Measure*) | Сохранение ≥ 97% качества линейного зонда (точность `RidgeClassifier` для классов, \|Spearman\| косинусной близости для STS) относительно полного 384-мерного вектора. |
| **Селектор** (*Selection Mapping*) | 1) Эмпирический кросс-валидационный перебор (`--empirical`).<br>2) Быстрый инференс через поставляемый классификатор `RandomForestClassifier` (`meta_model.skops`). |

Пакет **не переносит** тяжеловесные мета-признаки комбинаторных задач (SAT/TSP) и конвейеры вроде AutoFolio: текстовые эмбеддинги требуют компактного профилирования геометрии, рассчитываемого за доли секунды.

---

## Оценка внутренней размерности данных (Intrinsic Dimension)

Для анализа сжатия многообразий пакет опирается на методы оценки топологической и линейной размерности (Bac et al., 2021):

- **Линейная оценка (`intrinsic_dim_pca_95`):** количество главных компонент, покрывающих 95% дисперсии матрицы данных (соответствует метрике T3 в таксономии Lorena и оценке PCA-ratio в пакете `scikit-dimension`).
- **Фрактальная оценка (`intrinsic_dim_twonn`):** внутренняя размерность многообразия по алгоритму TwoNN (Facco et al., 2017), оцениваемая по отношению расстояний до первого и второго ближайших соседей точки.
  - Реализована автономная fallback-версия на базе `sklearn.neighbors.NearestNeighbors`, не требующая внешних C-зависимостей вроде `dadapy` или `scikit-dimension`. При `n < 5` метод безопасно возвращает `0.0`.

> **Геометрическая интерпретация:** один порог PCA не описывает нелинейную геометрию данных. Существенное расхождение между оценками PCA-95 (глобальная линейная емкость) и TwoNN (локальная топологическая размерность) служит прямым индикатором сильной нелинейной кривизны многообразия эмбеддингов.

Общая таксономия мета-признаков согласуется с фундаментальным обзором **Rivolli et al. (2022)** (*Meta-features for meta-learning*). Набор признаков оптимизирован именно для плотных представлений трансформеров, исключая дорогостоящие табличные статистики.

Воспроизводимость: обучение мета-модели и runtime-инференс используют единую функцию `build_meta_feature_vector` из `src/dataset_complexity_profiler/features.py`.

---

## Меры сложности классификации (Lorena et al.)

В исследовательском пайплайне (`research/feature_selection.ipynb`) анализировались меры сложности задач классификации по таксономии **Lorena et al. (2019/2021)** и библиотеке `PyMFE` (на базе ECoL / Ho & Basu):

| Группа мер (Lorena) | Примеры признаков PyMFE | Геометрический смысл в эмбеддингах |
| :--- | :--- | :--- |
| **Перекрытие признаков** (*Feature Overlap*) | `f1.mean`, `f2.mean`, `f3.mean` | Максимальная и средняя различительная сила отдельных координат эмбеддинга. |
| **Граница и окрестность** (*Neighborhood*) | `n1`, `n2.mean`, `n3.mean`, `lsc` | Доля пограничных точек и локальная однородность классов («враждебные» соседи). |
| **Линейная разделимость** (*Linearity*) | `l1.mean`, `l2.mean` | Ошибка линейного классификатора (SVM/Ridge) на исходном пространстве. |
| **Размерность** (*Dimensionality*) | `t2`, `t3`, `t4` | Отношение числа объектов к размерности (`n / d`) и плотность упаковки. |
| **Топология сети** (*Network*) | `density`, `cls_coef`, `hubs.mean` | Свойства графа ε-близости (вычислительно нецелесообразно на 384 измерениях). |

**Результат отбора признаков:** добавление 30 расширенных признаков PyMFE снижало обобщающую способность мета-модели на кросс-валидации относительно компактной пятерки базовых признаков. В runtime библиотеки PyMFE **не импортируется**, экономя время выполнения и память.

---

## Специализация и отличия пакета

1. **Эмбеддинги вместо таблиц:** все мета-признаки рассчитываются в семантическом пространстве `SentenceTransformer`, а не по сырым колонкам разнородных таблиц.
2. **Landmarking через зонды:** концепция *landmarking* (Pfahringer et al., 2000) реализована через быстрые линейные зонды (`RidgeClassifier` для классификации и ранговая корреляция `|Spearman|` косинусной близости для STS), определяющие разделимость пространства до запуска тяжелых вычислений.
3. **Консервативный прогноз порядка величины:** мета-модель предсказывает не точный гиперпараметр классификатора, а консервативную границу сжатия (степень двойки) — бакет сетки PCA, а не выбор классификатора `scikit-learn`.

Метрики валидации и абляционные тесты: [`META_MODEL_METRICS.md`](META_MODEL_METRICS.md).
Архитектура пайплайна и контракт модулей: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Литература

1. **Rice, J. R. (1976).** The Algorithm Selection Problem. *Advances in Computers*, 15, 65–118.
2. **Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019).** Automated Algorithm Selection: Survey and Perspectives. *Evolutionary Computation*, 27(1), 3–45.
3. **Lorena, A. C., Garcia, L. P., Lehmann, J., de Souto, M. C., & Ho, T. K. (2019).** How Complex Is Your Classification Problem? A Survey on Measuring Classification Complexity. *ACM Computing Surveys*, 52(5), 1–34. [arXiv:1808.03591](https://arxiv.org/abs/1808.03591).
4. **Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017).** Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7(1), 12140. [DOI:10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y).
5. **Bac, J., Mirkes, E. M., Gorban, A. N., Tyukin, I., & Zinovyev, A. (2021).** Scikit-dimension: a Python package for intrinsic dimension estimation. *Entropy*, 23(10), 1368. [arXiv:2109.02596](https://arxiv.org/abs/2109.02596).
6. **Rivolli, A., Garcia, L. P., Soares, C., Vanschoren, J., & de Carvalho, A. C. (2022).** Meta-features for meta-learning: a review and categorization. *Knowledge-Based Systems*, 240, 108101. [DOI:10.1016/j.knosys.2021.108101](https://doi.org/10.1016/j.knosys.2021.108101).
7. **Pfahringer, B., Bensusan, H., & Giraud-Carrier, C. (2000).** Meta-Learning by Landmarking Various Learning Algorithms. *Proceedings of the 17th International Conference on Machine Learning (ICML)*, 743–750.

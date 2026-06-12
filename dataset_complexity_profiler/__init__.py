"""Пакет dataset_complexity_profiler.

Предоставляет основной класс `DatasetProfiler` для анализа текстовых датасетов
и рекомендации оптимальной размерности эмбеддингов.
"""

from .dataset_adapter import DatasetProfiler

__all__ = ["DatasetProfiler"]

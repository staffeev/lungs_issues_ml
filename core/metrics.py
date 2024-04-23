"""Модуль с доступными для подсчёта метриками.

Для расширямости и масштабируемости все виды метрик будут 
объявлены или определены здесь вне зависимости от их происхождения."""
from torcheval.metrics import *


__all__ = [
    'get_metrics'
]


def get_metrics(metrics_name: list[str]):
    return list(eval(name)() for name in metrics_name)


def accuracy_score() -> Metric:
    """Возвращаёт объект accuracy-метрики. """
    return MulticlassAccuracy(num_classes=3)


def precision_score() -> Metric:
    """Возвращает объект precision-метрики. """
    return MulticlassPrecision(num_classes=3)


def recall_score() -> Metric:
    """Возвращает объект recall-метрики. """
    return MulticlassRecall(num_classes=3)


def f1_score() -> Metric:
    """Возвращает объект f1-score-метрики. """
    return MulticlassF1Score(num_classes=3)

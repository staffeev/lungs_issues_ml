"""
Модуль с доступными для подсчёта метриками.

Для расширямости и масштабируемости все виды метрик будут 
объявлены или определены здесь вне зависимости от их происхождения.

"""
from torch.nn import Module
from torch.utils.data import DataLoader
from torcheval.metrics import Metric
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision
from torcheval.metrics import MulticlassRecall, MulticlassF1Score

__CLASSES_NUM = 3


def accuracy_score(model: Module, dataloader: DataLoader) -> float:
    """Считает accuracy модели на выборке.
    
    Аргументы:
    - model: Module - модель, предсказания которой оцениваются.
    - dataloader: Dataloader - загрузчик данных, который загружает выборку."""

    metric = MulticlassAccuracy(num_classes=__CLASSES_NUM)
    return count_metric(model, dataloader, metric)


def precision_score(model: Module, dataloader: DataLoader) -> float:
    """Считает precision модели на выборке.
    
    Аргументы:
    - model: Module - модель, предсказания которой оцениваются.
    - dataloader: Dataloader - загрузчик данных, который загружает выборку."""

    metric = MulticlassPrecision(num_classes=__CLASSES_NUM)
    return count_metric(model, dataloader, metric)


def recall_score(model: Module, dataloader: DataLoader) -> float:
    """Считает recall модели на выборке.
    
    Аргументы:
    - model: Module - модель, предсказания которой оцениваются.
    - dataloader: Dataloader - загрузчик данных, который загружает выборку."""

    metric = MulticlassRecall(num_classes=__CLASSES_NUM)
    return count_metric(model, dataloader, metric)


def f1_score(model: Module, dataloader: DataLoader) -> float:
    """Считает f1_score модели на выборке.
    
    Аргументы:
    - model: Module - модель, предсказания которой оцениваются.
    - dataloader: Dataloader - загрузчик данных, который загружает выборку."""

    metric = MulticlassF1Score(num_classes=__CLASSES_NUM)
    return count_metric(model, dataloader, metric)


def count_metric(model: Module, dataloader: DataLoader, metric: Metric) -> float:
    """Утилитная функция для уменьшения дублирования кода.
    Считает метрику модели на данной выборке и возвращает результат.
    
    Аргументы:
    - model: Module - модель, предсказания которой оцениваются.
    - dataloader: Dataloader - загрузчик данных, который загружает выборку.
    - metric: Metric - метрика, которая считается по полученным данным.
    """
    model.eval()
    for X, y in dataloader:
        X, y = X, y
        metric.update(model(X), y)
    return metric.compute().item()


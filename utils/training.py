import os
from tqdm import tqdm
from torch import save
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.optimizer import Optimizer
from typing import Optional
from torcheval.metrics import Metric


def run_epoch(model: Module, loss_function: Module, dataloader: DataLoader, 
              metrics: list[Metric], optimizer: Optional[Optimizer]=None,
              description: str='?????') -> tuple[int, list[float]]:
    """Запускает одну (train/valid) эпоху. Возвращет итоговый loss и значения метрик.

    Аргументы:
    - model: Module - запускаемая модель.
    - loss_function: Module - функция потерь.
    - dataloader: DataLoader - загрузчик данных.
    - metrics: list[Metric] - список метрик, необходмых для подсчёта.
    - optimiser: Optional[Optimizer]=None - оптимизатор модели. Если не передан, то модель не обучается.
    - description: str = "?????" - описание текущей эпохи. """

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0
    for data, target in tqdm(dataloader, description):
        predicts = model(data)
    
        for metric in metrics:
            metric.update(predicts, target)
        
        loss = loss_function(predicts, target)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()


    return total_loss / len(dataloader), list(map(lambda metric: metric.compute().item(), metrics))
    

def train_model(model: Module, loss_function: Module, optimizer: Optimizer, 
                dataset: Dataset, train_valid_fractions: list[float],
                num_epochs: int, batch_size: int, model_title: str,
                metrics: list[Metric], save_path: str):
    """Обучает модель, то есть минимизирует лосс на конкретном датасете.
    
    Аргументы:
    - model: Module - обучаемая модель.
    - loss_function: Module - функция потерь.
    - optimizer: Optimizer - оптимизатор модели.
    - dataset: Dataset - данные. Будут разделены на train и valid.
    - train_valid_fractions: list[float] - в каком отношении делить выборку.
    - num_epochs: int - количество эпох, которые будет обучаться модели.
    - batch_size: int - размер пачки (батча).
    - model_title: str - под каким именем сохранять файлы состояния.
    - metrics: list[Metric] - список метрик для подсчета.
    - save_path: bool - путь, где нужно сохранять состояния модели. """

    train_dataset, valid_dataset = random_split(dataset, train_valid_fractions)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    train_losses, train_metrics = [], []
    valid_losses, valid_metrics = [], []

    for i in range(num_epochs):
        train_loss, train_metric_values = run_epoch(model=model, 
                                                    loss_function=loss_function, 
                                                    dataloader=train_dataloader, 
                                                    metrics=metrics, 
                                                    optimizer=optimizer, 
                                                    description=f"{model_title}. Train epoch #{i}")
        train_losses.append(train_loss)
        train_metrics.append(train_metric_values)

        valid_loss, valid_metric_values = run_epoch(model=model, 
                                                    loss_function=loss_function, 
                                                    dataloader=valid_dataloader, 
                                                    metrics=metrics, 
                                                    description=f"{model_title}. Valid epoch #{i}")
        
        valid_losses.append(valid_loss)
        valid_metrics.append(valid_metric_values)

        save(model.state_dict(), os.path.join(save_path, f'{model_title}[{i}].pt'))
        save(optimizer.state_dict(), os.path.join(save_path, f'{model_title}_optim[{i}].pt'))
    

    


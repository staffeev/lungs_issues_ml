## TODO Иван, решай сам, что тут норм, а чет убрать.
from torch.nn import Module
from torch.utils.data import DataLoader
import csv
from core import ImageDataset
from tqdm import tqdm

# TODO: что делать с device.
# device = device("cuda") if cuda.is_available() else device("cpu")

__all__ = [
    'get_and_save_predicts'
]


def get_and_save_predicts(model: Module, dataloader: DataLoader, save_path: str):
    """Получить предсказания модели и сохранить их по файлу.
    
    Аргументы:
    - model: Module - модель, дающая предсказания.
    - dataloader: DataLoader - загрузчик данных. 
    - save_path: str - путь, по которому нужно сохранить. """
    
    predicts = get_predicts(model, dataloader)
    save_predicts(predicts, save_path)


def get_predicts(model: Module, dataloader: DataLoader) -> list[tuple[int, int]]:
    """Получает предсказания модели для тестовой выборки. 
    Возвращает список предсказаний. Элементом является пара чисел: (id, pred)
    
    Аргументы:
    - model: Module - модель, предсказания которой и получают.
    - dataloader: DataLoader - загрузчик данных с тестовой выборки. """
    model.eval()
    predicts = []
    for i, image in enumerate(tqdm(dataloader, desc="Getting predictions")):
        image = image.to() #device)
        predict = model(image).argmax(1).item()
        predicts.append((i, predict))
    return predicts


def save_predicts(predicts: list[tuple[int, int]], path: str) -> None:
    """Сохраняет предсказания по данному пути.

    Аргументы:
    - predicts: list[tuple[int, int]] - сами предсказания.
    - path: str - путь, по которому нужно сохранить. """

    with open(path, "w", encoding="utf-8", newline="") as file:
        wr = csv.writer(file, delimiter=",")
        wr.writerow(["id", "target_feature"])
        wr.writerows(predicts)

"""Наш кастомный датасет."""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform, Identity


__all__ = [
    'ImageDataset'
]


class ImageDataset(Dataset):
    """Кастомный датасет под нашу задачу. 
    В отличие от torch-а, загружает изображения из папки, не разделённую на классы-директории. 
    Если данные размечены, возвращает с ответом, если нет, то просто изображение. """

    def __init__(self, image_dir: str, labels_file=None, transform: Transform = None):
        """Инициализирует датасет.
        
        Аргументы:
        - image_dir: str - путь к папке с изображениями.
        - labels_file: str = None - путь к csv-файлу с разметкой.
          Если объект не передан, данные считаются неразмеченными.
        - transform: Transform = None - объект трансформер. По умолчанию Identity. """
        self.image_dir = image_dir

        if labels_file:
            self.labels = pd.read_csv(labels_file)
        else:
            self.labels = None

        if transform is not None:
            self.transform = transform 
        else:
            self.transform = Identity()

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(os.listdir(self.image_dir))
    
    def __getitem__(self, index: int):
        image_path = os.path.join(self.img_dir, f"img_{index}.png")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if self.labels:
            return image, self.labels.iloc[index, 1]

        return image
        

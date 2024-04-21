"""Пакет с доступными для обучения моделями.

Для расширямости и масштабируемости все модели будут объявлены здесь. """
from .cnn import *
from .unet import *
from .vgg16 import *
from .resnet import *

__all__ = [
    'CNN', 'ResNet', 'UNet', 'VGG16', 'VGG19',
]

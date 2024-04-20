"""Пакет с доступными для обучения моделями.

Для расширямости и масштабируемости все модели будут объявлены здесь. """
from .cnn import *
from .unet import *
from .vggnet import *


__all__ = [
    'CNN', 'UNet', 'VGGNet'
]

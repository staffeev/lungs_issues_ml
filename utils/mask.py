"""Модуль для работы с масками и прочее. (Не используется, но вдруг). """
from PIL import Image


__all__ = [
    'apply_mask'
]


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Применяет к изображению маску и возвращает результат.
    
    Аргументы:
    - image: Image - исходное изображение.
    - mask: Image - маска, которую нужно применить. """
    return Image.composite(image, mask, mask)
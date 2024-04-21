'''Модуль с загрузчиком трансформеров.'''
from torchvision.transforms.v2 import *


__all__ = [
    'get_transform'
]


def get_transform(transforms_info: list = None) -> Transform:
    """По данному списку создаёт объект трансформации.

    Аргументы:
    - transforms_info: list = None - список с описанием.
      Должен состоять из словарей ('name': <имя класса>, 'args': <словарь аргументов>).
      Если значение не передано, или список пустой возвращает тождественное преобразование. """
    if not transform_info:
        return Identity()
    
    transforms = []
    for transform_info in transforms_info:
        name = transform_info['name']
        args = transform_info['args']
        transforms.append(create_transform(name, args))    

    return Compose(transforms)


def create_transform(class_name: str, kwargs: dict) -> Transform:
    """Создаёт объект трансформер.
    
    Аргументы:
    - class_name: str - название класса трансформации.
    - kwargs: dict - параметры трансформации. """

    class_object =  eval(f'{class_name}')
    return class_object(**kwargs)

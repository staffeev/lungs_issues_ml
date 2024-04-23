"""Модуль с функциями для рисования графиков. """
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.axes import Axes


__all__ = [
    'plot_graphs',
]


def plot_graphs(axes: Axes, *args, **kwargs):
    """Рисует графики данных по заданным осям. Вызывает plot.
    
    Аргументы:
    - axes: Axes - оси, по которым строится график.
    - *args - данные для построения графика.
    - **kwargs - дополнительные аргументы. """
    return axes.plot(*args, **kwargs)


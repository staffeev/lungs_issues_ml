"""Полезные блоки и функции, которые будут использованы в других моделях"""
from torch import nn


class SkipConnection(nn.Module):
    '''SkipConnection - класс для создания SkipConnection.
    
    Просто соединяет начало и конец между заданным куском сети.'''

    def __init__(self, *args: nn.Module, sampling=None):
        """ Метод инициализирует блок.

        Аргументы:
        - net - та часть, сети, которая между skip-connection
        - sampling - преобразования для совпадения размерностей.
        """

        super().__init__()

        self.net = nn.Sequential(*args)
        if sampling is None:
            sampling = nn.Identity()

        self.downsample = sampling

    def forward(self, X):
        identity = self.downsample(X)

        # F(x) + x
        return self.net(X) + identity

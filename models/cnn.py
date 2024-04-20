"""Простая реализация CNN."""

from torch import nn


__all__ = [
    'CNN'
]


class CNN(nn.Module):
    """CNN (Convolutional Neural Network) - свёрточная нейронная сеть.
    
    Архитектура сети:
    1) Четыре блока свёртки (Conv2d -> BathcNorm -> MaxPoll2d -> ReLU)
    2) Классификатор - полносвязная сеть из входного, трех скрытых, выходного слоёв.
       Структура слоя: (Linear -> ReLU -> Dropout). """
    def __init__(self, in_channels=3):
        """Метод инициализирует модель.

        Аргументы:
        - in_channels: int = 3 - количество входных каналов изображения.
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, 6, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(15488, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 3),
        )
    
    def forward(self, x):
        x =  self.conv_layers(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
    
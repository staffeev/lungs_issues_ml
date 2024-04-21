"""Реализация VGG-16"""
import torch.nn as nn
from .utils import SkipConnection

__all__ = [
    'VGG16'    
]


class VGG16(nn.Module):
    """VGG16 - глубокая свёрточная нейронная сеть. Разновидность VGG.
    
    Архитектура сети:
    - Много свёрточных слоёв и в конце классификатор.
    - Для избежания проблем с обучением добавил множество skip-connection. """
    def __init__(self, in_channels: int=3):
        """Метод инициализирует модель.

        Аргументы:
        in_channels: int = 3 - количество входных каналов. """
        super().__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(),
            SkipConnection(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(),
            SkipConnection(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(),
            ),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(),
            SkipConnection(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(),
            SkipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2),

            SkipConnection(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(),
                SkipConnection(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                )
            ),
            nn.MaxPool2d(2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Dropout(),
            SkipConnection(
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
            ),
            nn.Linear(4096, 3)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

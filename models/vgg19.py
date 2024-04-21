import torch.nn as nn
from .utils import SkipConnection


__all__ = [
    'VGG19'
]

class VGG19(nn.Module):
    """VGG16 - глубокая свёрточная нейронная сеть, разновидность VGG.
    
    Архитектура сети:
    - Много свёрточных слоёв и в конце классификатор.
    - Для избежания проблем с обучением добавил множество skip-connection. """
    def __init__(self, in_channels: int=3) -> None:
        """Метод инициализирует модель.

        Аргументы:
        in_channels: int = 3 - количество входных каналов. """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SkipConnection(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            SkipConnection(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            SkipConnection(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                SkipConnection(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            SkipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                SkipConnection(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            SkipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                SkipConnection(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 3),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
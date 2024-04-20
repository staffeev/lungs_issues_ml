"""Реализация модели UNet. 
Предпологалась, что она будет использоваться для сегментации, но что-то не сраслось."""
import torch
from torch import nn


__all__ = [
    'UNet'
]


class UNet(nn.Module):
    """U-Net - свёрточная нейронная сеть, предназначенная для задач сегментации.
    
    Архитектура сети:
    - Сеть содержит сверточную (слева) и разверточную части (справа), 
      поэтому архитектура похожа на букву U, что и отражено в названии. 
      На каждом шаге количество каналов признаков удваивается (на первом 64). """
    
    def __init__(self, in_channels: int=3, out_channels: int=1, bilinear: bool=True):
        """Метод инициализирует модель.

        Аргументы:
        - in_channels: int = 3 - количество входных каналов изображения.
        - out_channels: int = 1 - количество выходных каналов. Определяет на сколько классов сегментировать. """

        super().__init__()

        self.incomming = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, out_channels)


    def forward(self, x):
        x1 = self.incomming(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def use_checkpointing(self):
        self.incomming = torch.utils.checkpoint(self.incomming)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class DoubleConv(nn.Module):
    """ DoubleConv - составной-блок U-Net и других частей. Просто двойное применение свёртки.
    
    Архитектура блока:
    - Каждая свёртка имеет вид: Conv -> BatchNorm -> ReLU. """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int=None):
        """Метод инициализирует модель.

        Аргументы:
        - in_channels: int = 1 - количество входных каналов изображения.
        - mid_channels: int = out_channels - количество каналов в среднем слое.
        - out_channels: int = 1 - количество выходных каналов. Определяет на сколько классов сегментировать. """

        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Down - составной блок левой части U-Net. Сворачивает значения. 
    
    Архитектура блока: 
    - MaxPoll -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        """Метод инициализирует модель.

        Аргументы:
        - in_channels: int - количество входных каналов изображения.
        - out_channels: int - количество выходных каналов. """
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up - составной блок правой части U-Net. Развёртывает значения, а также принимает с предыдущего. 
    
    Архитектура блока:
    - Upsample/ConvTranspose2d -> DoubleConv. """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool=True):
        """Метод инициализирует модель.

        Аргументы:
        - in_channels: int - количество входных каналов изображения.
        - out_channels: int - количество выходных каналов. 
        - bilinear: bool = True - определяет метод для Upsampling. """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.functional.F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """ OutConv - составной блок U-Net. Простая свёртка. 
    Если добавиться пост-обработка результата, то оно будет применяться здесь. """

    def __init__(self, in_channels: int, out_channels: int):
        """Метод инициализирует модель.

        Аргументы:
        - in_channels: int - количество входных каналов.
        - out_channels: int - количество выходных каналов. """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
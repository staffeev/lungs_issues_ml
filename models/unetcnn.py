import torch
from torch import nn

__all__ = [
    'UNetCNN'
]


from models.unet import UNet
from models.simple_cnn import CNN


class UNetCNN(nn.Module):
    def __init__(self, in_channels: int=3):
        super().__init__()

        self.unet = UNet(in_channels, 32)
        self.cnn = CNN(32)


    def forward(self, x):
        x = self.unet(x)
        x = self.cnn(x)
        return x



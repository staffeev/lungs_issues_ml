import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 256 x 256 x 1 -> 256 x 256 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # -> 256 x 256 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),       # -> 128 x 128 x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> 128 x 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# -> 128 x 128 x 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),          # -> 64 x 64 x 128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# -> 64 x 64 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# -> 64 x 64 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# -> 64 x 64 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)         # -> 32 x 32 x 256
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 2),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = VGGNet()

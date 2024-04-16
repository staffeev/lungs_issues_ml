import torch.nn as nn


class block(nn.Module):
    def __init__(self, *args, downsample=None):
        """
        block.__init__(self, net, downsample)

        Аргументы:
        - net - та часть, сети, которая между skip-connection
        - downsample - преобразования к X, чтобы он совпал.
        """

        super().__init__()

        self.net = nn.Sequential(*args)
        if downsample is None:
            downsample = nn.Identity()

        self.downsample = downsample

    def forward(self, X):
        identity = self.downsample(X)

        # F(x) + x
        return self.net(X) + identity


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            block(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            block(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            block(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            block(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.BatchNorm2d(128),
            block(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                block(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            ),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Dropout(),
            block(
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
            ),
            nn.Linear(4096, 3)
        )
        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
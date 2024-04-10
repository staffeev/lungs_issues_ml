from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
            
        )
    
    def forward(self, x):
        x = self.seq2(self.seq1(x))
        x = x.view(x.size(0), -1)
        x = self.seq3(x)
        return x

# class FullyConnectedModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(5 * 5, 3),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.seq(x.reshape(-1, 5 * 5))
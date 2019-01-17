import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(64, 128, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(128, 256, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.dense = nn.Sequential(
            nn.Linear(256 * 16 * 16, 256 * 2 * 2),
            nn.ReLU(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        y = self.conv(img)
        y = y.reshape(img.shape[0], -1)
        y = self.dense(y)
        return y

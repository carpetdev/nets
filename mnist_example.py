# A simple feed-forward convolutional network adapted from the example given in PyTorch's "Blitz"; ReLU activators finishing with softmax, MSE loss, SGD update

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets.mnist as mnist

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # Suppose input is single channel, 28x28 pixels, padded to 32x32
        # 6 feature maps, 28x28 pixels, max-pooled to 14x14
        # then 16 feature maps, 12x12 pixels, max-pooled to 6x6, reshaped to a 16*6*6 row vector
        # followed by 3 fully connected layers 16*6*6 -> 120 -> 84 -> 10
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Note dimensions are sample, channel, row, column
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)

net = Net()

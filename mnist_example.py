# A simple feed-forward convolutional network adapted from the example given in PyTorch's "Blitz"; ReLU activators finishing with softmax, MSE loss, SGD update
# This program has been christened Lucida, or Lucy for short!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision.transforms as T
import torch.optim as optim
from torchvision.datasets import MNIST

class Lucida(nn.Module):

    def __init__(self):
        # Suppose input is single channel, 32x32 pixels
        # 6 feature maps, 28x28 pixels, max-pooled to 14x14
        # then 16 feature maps, 12x12 pixels, max-pooled to 6x6, reshaped to a 16*6*6 row vector
        # followed by 3 fully connected layers 16*6*6 -> 120 -> 84 -> 10
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Note dimensions are sample, channel, row, column (given by DataLoader)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.softmax(dim=1)

def train_MNIST(model):
        # DataLoader produces an iterable with a tuples of (data, label).
        batch_size = 4
        learning_rate = 0.1
        epochs = 2

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        transform = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize(0.5, 0.5)])
        train_data = MNIST('./assets', train=True, download=True, transform=transform)
        train_loader = D.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = torch.zeros(batch_size, 10, device=device).scatter(1, labels.reshape(batch_size,1), 1) # One-hot encoding
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if i % 2000 == 1999:
                    print(f"[{epoch+1:d}, {i+1:d}] loss = {loss.item()}")
                    # # print(outputs)
                    # # print(labels)


lucida = Lucida()
train_MNIST(lucida)

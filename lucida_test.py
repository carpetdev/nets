import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision.transforms as T
import torch.optim as optim
from torchvision.datasets import MNIST
from matplotlib import pyplot

class Lucida(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.softmax(dim=1)

lucy = torch.load("./models/lucida_trained.pt")

transform = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize(0.5, 0.5)])
test_data = MNIST("./assets", train=True, download=True, transform=transform)
test_loader = D.DataLoader(test_data)

for test_input, label in iter(test_loader):
    pyplot.imshow(test_input[0,0,:,:], cmap='gray')
    pyplot.show(block=False)
    hi = input("Press Enter to test")
    print(lucy(test_input).argmax().item())
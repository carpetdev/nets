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
test_data = MNIST("./assets", train=False, download=True, transform=transform)
test_loader = D.DataLoader(test_data)

wrong_tests = 0
for i, data in enumerate(test_loader):
    test_input, label = data
    # # pyplot.imshow(test_input[0,0,:,:], cmap='gray')
    # # pyplot.show(block=False)
    # # input("Press Enter to test")
    # # print(lucy(test_input).argmax().item())
    if lucy(test_input).argmax().item() != label:
        wrong_tests += 1
    if i % 1000 == 999:
        print(f"There have been {wrong_tests} incorrect tests.")
print(f"There were {wrong_tests} incorrect tests.")
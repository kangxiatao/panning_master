'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, c=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(c, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 1*28*28 => 6*24*24
        out = F.max_pool2d(out, 2)  # 6*24*24 => 6*12*12
        out = F.relu(self.conv2(out))  # 6*12*12 => 16*8*8
        out = F.max_pool2d(out, 2)  # 6*8*8 => 16*4*4
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import squeezenet1_1


class BeaconCNN(nn.Module):
    """
    NN model specifically for beacon mini game
    """
    def __init__(self, *args):
        super(BeaconCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.conv4 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=8, dilation=8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv3(x)
        return x


class FeatureCNN(nn.Module):
    """
    CNN model based on feature inputs
    """
    def __init__(self, n_features):
        super(FeatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_features, 12, kernel_size=3, stride=1, padding=1)  # TODO: Should be 3 to preserve size
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FeatureCNNFC(nn.Module):
    """
    CNN model based on feature inputs
    """
    def __init__(self, n_features, screen_size=64):
        super(FeatureCNNFC, self).__init__()
        self.conv1 = nn.Conv2d(n_features, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.fc2 = nn.Linear(screen_size ** 2, screen_size ** 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 ** 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = squeezenet1_1(pretrained=True)
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    x = torch.tensor(np.random.random((64, 1, 1, 1)))
    b = BeaconCNN()
    print(b(x.float()).squeeze().shape)
    pass

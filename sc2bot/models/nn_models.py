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
        self.name = 'BeaconCNN'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv3(x)
        return x


class BeaconCNN2(nn.Module):
    """
    NN model specifically for beacon mini game
    """
    def __init__(self, *args):
        super(BeaconCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)
        self.name = 'BeaconCNN'

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
        self.conv1 = nn.Conv2d(n_features, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)
        self.name = f'FeatureCNN{n_features}'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FeatureCNNFCBig(nn.Module):
    """
    CNN model based on feature inputs
    """
    def __init__(self, n_features, screen_size=64):
        super(FeatureCNNFCBig, self).__init__()

        self.conv1 = nn.Conv2d(n_features, 12, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.name = f'FeatureCNNFCBig{n_features}'
        self.screen_size = screen_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.screen_size ** 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FeatureCNNFCLimited(nn.Module):
    """
    CNN model based on feature inputs
    """
    def __init__(self, n_features, radius, screen_size=64):
        super(FeatureCNNFCLimited, self).__init__()
        self.conv1 = nn.Conv2d(n_features, 6, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(12 * int(screen_size / 4) ** 2, int(screen_size / 4) ** 2)
        self.fc2 = nn.Linear(int(screen_size / 4) ** 2, radius ** 2)
        self.name = f'FeatureCNNFC{n_features}'
        self.screen_size = screen_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 12 * int(self.screen_size/4) ** 2)
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

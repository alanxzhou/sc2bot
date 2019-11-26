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
        super(BeaconCNN, self).__init__(*args)
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


if __name__ == '__main__':
    model = squeezenet1_1(pretrained=True)
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

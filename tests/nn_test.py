import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class NN(nn.Module):

    def __init__(self, dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(dim, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ToyData(Dataset):

    def __init__(self, transform=None):
        self.data = [np.ones(10) * i for i in range(5) for _ in range(1000)]
        self.labels = [i for i in range(5) for _ in range(1000)]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        label = self.labels[idx]
        # sample = {'x': x, 'y': y}
        if self.transform:
            sample = self.transform(sample)

        return sample, label


class ToTensor(object):

    def __call__(self, sample):
        return torch.from_numpy(sample)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # Loading in Data:
    classes = np.arange(0, 5)
    toy = ToyData(transform=ToTensor())
    trainloader = DataLoader(toy, batch_size=16, shuffle=True)
    # for i_batch, sample_batched in enumerate(trainloader):
    #     print(i_batch, sample_batched)
    #     break

    # Setting up net:
    net = NN(10).double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('training...')
    for epoch in range(100):
        print(f'epoch {epoch}')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

    print('Finished Training')

    # Now we evaluate
    testloader = DataLoader(toy, batch_size=16, shuffle=False)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

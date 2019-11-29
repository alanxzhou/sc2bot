import copy
import os
from collections import deque

from pysc2.agents.base_agent import BaseAgent
from sc2bot.utils.epsilon import Epsilon
from sc2bot.utils import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim


class BaseRLAgent(BaseAgent):

    def __init__(self):
        super(BaseRLAgent, self).__init__()
        self.training = False
        self.max_frames = 10000000
        self._epsilon = Epsilon(start=1.0, end=0.1, update_increment=0.0001)
        self.gamma = 0.99
        self.train_q_per_step = 4
        self.train_q_batch_size = 256
        self.steps_before_training = 10000
        self.target_q_update_frequency = 10000

        self._Q_weights_path = "./data/test.pth"
        self._Q = BeaconCNN()
        if os.path.isfile(self._Q_weights_path):
            self._Q.load_state_dict(torch.load(self._Q_weights_path))
        self._Qt = copy.deepcopy(self._Q)
        self._Q.cuda()
        self._Qt.cuda()
        self._optimizer = optim.Adam(self._Q.parameters(), lr=1e-8)
        self._criterion = nn.MSELoss()
        self._memory = ReplayMemory(50000)

        self._loss = deque(maxlen=1000)
        self._max_q = deque(maxlen=1000)
        self._action = None
        self._screen = None
        self._fig = plt.figure()
        self._plot = [plt.subplot(2, 2, i+1) for i in range(4)]
        self._screen_size = 64
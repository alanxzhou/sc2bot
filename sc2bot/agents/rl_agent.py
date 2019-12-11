from abc import ABC, abstractmethod
import copy
from collections import deque
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from pysc2.lib import features
from sc2bot.utils.epsilon import Epsilon
from sc2bot.utils.replay_memory import ReplayMemory, Transition
from sc2bot.models.nn_models import BeaconCNN

import torch
import torch.nn as nn
import torch.optim as optim


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SELECT_POINT = actions.FUNCTIONS.select_point.id


class BaseRLAgent(BaseAgent, ABC):

    def __init__(self, save_name='./data/test.pth'):
        super(BaseRLAgent, self).__init__()
        self.training = False
        self.max_frames = 10000000
        self._epsilon = Epsilon(start=0.9, end=0.1, update_increment=0.0001)
        self.gamma = 0.99
        self.train_q_per_step = 4
        self.train_q_batch_size = 256
        self.steps_before_training = 10000
        self.target_q_update_frequency = 10000

        self.save_name = save_name
        self._Q = None
        self._Qt = None
        self._optimizer = None
        self._criterion = nn.MSELoss()
        self._memory = ReplayMemory(100000)

        self._loss = deque(maxlen=1000)
        self._max_q = deque(maxlen=1000)
        self.loss = []
        self.max_q = []
        self.reward = []
        self._action = None
        self._screen = None
        self._fig = plt.figure()
        self._plot = [plt.subplot(2, 2, i+1) for i in range(4)]
        self._screen_size = 64
        self.n_episodes = 0

    def initialize_model(self, model):
        self._Q = model
        self._Qt = copy.deepcopy(self._Q)
        self._Q.cuda()
        self._Qt.cuda()
        self._optimizer = optim.Adam(self._Q.parameters(), lr=1e-8)

    def load_model_checkpoint(self):
        self._Q.load_state_dict(torch.load(self.save_name + '.pth'))
        saved_data = pickle.load(open(f'{self.save_name}', 'rb'))
        self.loss = saved_data['loss']
        self.max_q = saved_data['max_q']
        self._epsilon._value = saved_data['epsilon']
        self.reward = saved_data['reward']
        self.n_episodes = saved_data['n_episodes']

    def get_env_action(self, action, obs, command=_MOVE_SCREEN):
        action = np.unravel_index(action, [1, self._screen_size, self._screen_size])
        target = [action[2], action[1]]
        # command = _MOVE_SCREEN  # action[0]   # removing unit selection out of the equation

        if command in obs.observation["available_actions"]:
            return actions.FunctionCall(command, [[0], target])
        else:
            return actions.FunctionCall(_NO_OP, [])

    def save_data(self, episodes_done=0):
        save_data = {'loss': self.loss,
                     'max_q': self.max_q,
                     'epsilon': self._epsilon._value,
                     'reward': self.reward,
                     'n_episodes': self.n_episodes}

        if episodes_done > 0:
            save_name = self.save_name + f'_checkpoint{episodes_done}'
        else:
            save_name = self.save_name
        torch.save(self._Q.state_dict(), save_name + '.pth')
        pickle.dump(save_data, open(f'{save_name}_data.pkl', 'wb'))

    def evaluate(self, env, max_episodes=10000, load_dict = True):
        if load_dict:
            self._Q.load_state_dict(torch.load(self.save_name + '.pth'))
        self._epsilon.isTraining = False
        self.run_loop(env, self.max_frames, max_episodes=max_episodes)

    def train(self, env, training=True, max_episodes=10000):
        self._epsilon.isTraining = training
        self.run_loop(env, self.max_frames, max_episodes=max_episodes)
        if self._epsilon.isTraining:
            self.save_data()

    @abstractmethod
    def run_loop(self, env, max_frames, max_episodes, evaluate_checkpoints):
        pass

    def get_action(self, s, unsqueeze=True):
        # greedy
        if np.random.rand() > self._epsilon.value():
            s = torch.from_numpy(s).cuda()
            if unsqueeze:
                s = s.unsqueeze(0).float()
            else:
                s = s.float()
            with torch.no_grad():
                self._action = self._Q(s).squeeze().cpu().data.numpy()
            return self._action.argmax()
        # explore
        else:
            action = 0
            target = np.random.randint(0, self._screen_size, size=2)
            return action * self._screen_size * self._screen_size + target[0] * self._screen_size + target[1]

    def train_q(self, squeeze=False):
        if self.train_q_batch_size >= len(self._memory):
            return

        s, a, s_1, r, done = self._memory.sample(self.train_q_batch_size)
        s = torch.from_numpy(s).cuda().float()
        a = torch.from_numpy(a).cuda().long().unsqueeze(1)
        s_1 = torch.from_numpy(s_1).cuda().float()
        r = torch.from_numpy(r).cuda().float()
        done = torch.from_numpy(1 - done).cuda().float()

        if squeeze:
            s = s.squeeze()
            s_1 = s_1.squeeze()

        # Q_sa = r + gamma * max(Q_s'a')
        Q = self._Q(s).view(self.train_q_batch_size, -1)
        Q = Q.gather(1, a)

        Qt = self._Qt(s_1).view(self.train_q_batch_size, -1)

        # double Q
        best_action = self._Q(s_1).view(self.train_q_batch_size, -1).max(dim=1, keepdim=True)[1]
        y = r + done * self.gamma * Qt.gather(1, best_action)
        # y = r + done * self.gamma * Qt.max(dim=1)[0].unsqueeze(1)

        # y.volatile = False
        # with y.no_grad():
        loss = self._criterion(Q, y)
        self._loss.append(loss.sum().cpu().data.numpy())
        self._max_q.append(Q.max().cpu().data.numpy().reshape(-1)[0])
        self._optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        self._optimizer.step()
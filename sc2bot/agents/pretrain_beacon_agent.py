import copy
from sc2bot.utils.replay_memory import ReplayMemory
from sc2bot.agents.beacon_agent import BeaconAgent
import pickle
import numpy as np

import torch
import time


class BeaconAgentPretrained(BeaconAgent):

    def __init__(self, save_name):
        super(BeaconAgentPretrained, self).__init__(save_name=save_name)

    def pretrain(self, memory_pretrained_fn, batch_size=512, iterations=int(1e5)):
        memory_pretrained = pickle.load(open(memory_pretrained_fn, 'rb'))
        self._memory = ReplayMemory(len(memory_pretrained))
        self._memory.memory = memory_pretrained
        self.train_q_batch_size = batch_size
        start_time = time.time()
        for i in range(iterations):
            if i % 500 == 0:
                # if i > 1:
                #     for key in self._Qt.state_dict():
                #         try:
                #             print(sum(self._Qt.state_dict()[key] - self._Q.state_dict()[key]))
                #         except TypeError as e:
                #             print(e)
                self._Qt = copy.deepcopy(self._Q)
            print(f'Training iteration {i}...', )
            self.train_q(squeeze=False)
            if i % 10000 == 0:
                torch.save(self._Q.state_dict(), f'{self.save_name}_{i}.pth')
                pickle.dump(self.loss, open(f'{self.save_name}_loss_{i}.pkl', 'wb'))
        end_time = time.time()
        print(f'Training completed. Took {start_time - end_time} seconds')
        torch.save(self._Q.state_dict(), f'{self.save_name}_{iterations}.pth')
        pickle.dump(self.loss, open(f'{self.save_name}_loss_{iterations}.pkl', 'wb'))

    def train_q(self, squeeze=False):
        if self.train_q_batch_size >= len(self._memory):
            return

        s, a, s_1, r, done = self._memory.sample(self.train_q_batch_size)
        s = torch.from_numpy(s).to(self.device).float()
        a = torch.from_numpy(a).to(self.device).long().unsqueeze(1)
        s_1 = torch.from_numpy(s_1).to(self.device).float()
        r = torch.from_numpy(r).to(self.device).float()
        done = torch.from_numpy(1 - done).to(self.device).float()

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

        temp_state_dict = self._Q.state_dict()

        # with y.no_grad():
        loss = self._criterion(Q, y)
        self.loss.append(loss.sum().cpu().data.numpy())
        self._max_q.append(Q.max().cpu().data.numpy().reshape(-1)[0])
        self._optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        self._optimizer.step()

        pass


if __name__ == '__main__':
    path = './data/'
    mapnames = ['MoveToBeacon']
    for mapname in mapnames:
        print(mapname)
        save_name = path + mapname + '_pretrain'
        memory_pretrained_fn = path + mapname + '/scripted_replaymemory_res32.pkl'
        pretrainer = BeaconAgentPretrained(save_name)
        pretrainer.pretrain(memory_pretrained_fn)

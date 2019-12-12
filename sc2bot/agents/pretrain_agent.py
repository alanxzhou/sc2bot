from sc2bot.utils.replay_memory import ReplayMemory
from sc2bot.agents.battle_agent import BattleAgent
import pickle

import torch
import time

class BattleAgentPretrained(BattleAgent):

    def __init__(self, save_name):
        super(BattleAgentPretrained, self).__init__(save_name=save_name)

    def pretrain(self, memory_pretrained_fn, batch_size=512, iterations=1000):
        memory_pretrained = pickle.load(open(memory_pretrained_fn, 'rb'))
        self._memory = ReplayMemory(len(memory_pretrained))
        self._memory.memory = memory_pretrained
        self.train_q_batch_size = batch_size
        start_time = time.time()
        for i in range(iterations):
            self.train_q()
        end_time = time.time()
        print(f'Training completed. Took {start_time - end_time} seconds')
        torch.save(self._Q.state_dic(), f'{self.save_name}_{iterations}.pth')


if __name__ == '__main__':
    path = './data/'
    mapnames = ['DefeatRoachesAntiSuicide', 'DefeatRoachesAntiSuicideMarineDeath0', 'DefeatRoaches']
    for mapname in mapnames:
        print(mapname)
        save_name = path + mapname + 'pretrain'
        memory_pretrained_fn = path + mapname +'/scripted_replaymemory.pkl'
        pretrainer = BattleAgentPretrained(save_name)
        pretrainer.pretrain(memory_pretrained_fn)
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
from sc2bot.models.nn_models import FeatureCNN, FeatureCNNFC, FeatureCNNFCBig
from sc2bot.agents.rl_agent import BaseRLAgent

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

_UNIT_TYPE = 6
_UNIT_HIT_POINTS = 8


class BattleAgentTotal(BaseRLAgent):
    """
    Agent where the entire army is selected
    """

    def __init__(self, save_name=None):
        super(BattleAgentTotal, self).__init__(save_name=save_name)
        self.initialize_model(FeatureCNNFCBig(3))
        self.steps_before_training = 5000
        # self.army_mean = None
        # self.army_reachable = 24

    # def get_action(self, s, unsqueeze=True):
    #     # greedy
    #     if np.random.rand() > self._epsilon.value():
    #         s = torch.from_numpy(s).cuda()
    #         if unsqueeze:
    #             s = s.unsqueeze(0).float()
    #         else:
    #             s = s.float()
    #         with torch.no_grad():
    #             self._action = self._Q(s).squeeze().cpu().data.numpy()
    #         return self._action.argmax()
    #     # explore
    #     else:
    #         action = np.random.randint(0, self.army_reachable ** 2)
    #         return action
    #
    # def get_env_action(self, action, obs, command=_MOVE_SCREEN):
    #     action = np.unravel_index(action, [self.army_reachable, self.army_reachable])
    #     y, x = (obs.observation["feature_screen"][_PLAYER_RELATIVE] == _PLAYER_FRIENDLY).nonzero()
    #     target = [int(action[1] - self.army_reachable/2 + round(x.mean())),
    #               int(action[0] - self.army_reachable/2 + round(y.mean()))]
    #     print('step')
    #     print(action[1], action[0])
    #     print(round(x.mean()), round(y.mean()), target[0], target[1])
    #     # target = [round(x.mean()), round(y.mean())]
    #     # command = _MOVE_SCREEN  # action[0]   # removing unit selection out of the equation
    #     z = np.array(target)
    #     friendly_coordinates = np.vstack((x, y)).T
    #     if bool(np.sum(np.all(z == friendly_coordinates, axis=1))):
    #         print('no action taken because we would attack our own unit')
    #         return actions.FunctionCall(_NO_OP, [])
    #
    #     if command in obs.observation["available_actions"] and target[0] >= 0 and target[1] >= 0:
    #         # if target
    #         return actions.FunctionCall(command, [[0], target])
    #     else:
    #         return actions.FunctionCall(_NO_OP, [])
    #         print(command)

    def run_loop(self, env, max_frames=0, max_episodes=10000, save_checkpoints=500):
        """A run loop to have agents and an environment interact."""
        total_frames = 0
        start_time = time.time()

        action_spec = env.action_spec()
        observation_spec = env.observation_spec()
        n_episodes = 0

        self.setup(observation_spec, action_spec)
        try:
            while n_episodes < max_episodes:

                obs = env.reset()[0]
                # remove unit selection from the equation by selecting the entire army on every new game.
                select_army = actions.FunctionCall(_SELECT_ARMY, [[False]])
                obs = env.step([select_army])[0]

                self.reset()

                while True:
                    total_frames += 1

                    screen_observations = obs.observation["feature_screen"][[_PLAYER_RELATIVE, _UNIT_TYPE, _UNIT_HIT_POINTS]]
                    s = np.expand_dims(screen_observations, 0)

                    if max_frames and total_frames >= max_frames:
                        print("max frames reached")
                        return
                    if obs.last():
                        print(f"Episode {n_episodes + 1}:\t total frames: {total_frames} Epsilon: {self._epsilon.value()}")
                        self._epsilon.increment()
                        break

                    action = self.get_action(s, unsqueeze=False)
                    env_actions = self.get_env_action(action, obs, command=_ATTACK_SCREEN)
                    obs = env.step([env_actions])[0]

                    r = obs.reward
                    s1 = np.expand_dims(obs.observation["feature_screen"][[_PLAYER_RELATIVE, _UNIT_TYPE, _UNIT_HIT_POINTS]], 0)
                    done = r > 0
                    if self._epsilon.isTraining:
                        transition = Transition(s, action, s1, r, done)
                        self._memory.push(transition)

                    if total_frames % self.train_q_per_step == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self.train_q(squeeze=True)

                    if total_frames % self.target_q_update_frequency == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self._Qt = copy.deepcopy(self._Q)

                    if not self._epsilon.isTraining and total_frames % 3 == 0:
                        a = 1

                n_episodes += 1
                if len(self._loss) > 0:
                    self.loss.append(self._loss[-1])
                    self.max_q.append(self._max_q[-1])
                if n_episodes % save_checkpoints == 0:
                    if n_episodes > 0:
                        self.save_data(episodes_done=n_episodes)

        except KeyboardInterrupt:
            pass
        finally:
            print("finished")
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


class BattleAgentTotalAttackOnly(BattleAgentTotal):
    def __init__(self, save_name=None):
        super(BattleAgentTotalAttackOnly, self).__init__(save_name=save_name)
        self.initialize_model(FeatureCNNFCBig(3))
        self.steps_before_training = 5000

    def get_action(self, s, unsqueeze=True):
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
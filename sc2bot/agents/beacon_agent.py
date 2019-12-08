import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import deque
from pysc2.agents.base_agent import BaseAgent

import torch
import torch.nn as nn
import torch.optim as optim

from sc2bot.agents.rl_agent import BaseRLAgent
from sc2bot.models.nn_models import BeaconCNN
from sc2bot.utils.epsilon import Epsilon
from sc2bot.utils.replay_memory import ReplayMemory, Transition

from pysc2.env import available_actions_printer
from pysc2.lib import actions
from pysc2.lib import features
import pickle

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


class BeaconAgent(BaseRLAgent):

    def __init__(self):
        super(BeaconAgent, self).__init__()
        self.initialize_model(BeaconCNN())

    @staticmethod
    def select_friendly_action(obs):
        player_relative = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        target = [int(friendly_x.mean()), int(friendly_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [[0], target])

    def run_loop(self, env, max_frames=0, max_episodes=10000):
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
                # remove unit selection from the equation by selecting the friendly on every new game.
                select_friendly = self.select_friendly_action(obs)
                obs = env.step([select_friendly])[0]
                # distance = self.get_reward(obs.observation["screen"])

                self.reset()

                while True:
                    total_frames += 1

                    self._screen = obs.observation["feature_screen"][5]
                    s = np.expand_dims(obs.observation["feature_screen"][5], 0)
                    # plt.imshow(s[5])
                    # plt.pause(0.00001)
                    if max_frames and total_frames >= max_frames:
                        print("max frames reached")
                        return
                    if obs.last():
                        print(f"Episode {n_episodes + 1}:\t total frames: {total_frames} Epsilon: {self._epsilon.value()}")
                        self._epsilon.increment()
                        break

                    action = self.get_action(s)
                    env_actions = self.get_env_action(action, obs)
                    obs = env.step([env_actions])[0]

                    r = obs.reward
                    s1 = np.expand_dims(obs.observation["feature_screen"][5], 0)
                    done = r > 0
                    if self._epsilon.isTraining:
                        transition = Transition(s, action, s1, r, done)
                        self._memory.push(transition)

                    if total_frames % self.train_q_per_step == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self.train_q()
                        # pass

                    if total_frames % self.target_q_update_frequency == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self._Qt = copy.deepcopy(self._Q)
                        # self.show_chart()
                        # pass

                    if not self._epsilon.isTraining and total_frames % 3 == 0:
                        # self.show_chart()
                        a = 1
                n_episodes += 1
                if len(self._loss) > 0:
                    self.loss.append(self._loss[-1])
                    self.max_q.append(self._max_q[-1])

        except KeyboardInterrupt:
            pass
        finally:
            print("finished")
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))
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
from sc2bot.models.nn_models import BeaconCNN, BeaconCNN2
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

    def __init__(self, save_name=None, load_name=None):
        super(BeaconAgent, self).__init__(save_name=save_name, load_name=None)
        self.initialize_model(BeaconCNN2())
        self.features = 5
        self.train_q_per_step = 4

    @staticmethod
    def select_friendly_action(obs):
        player_relative = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        target = [int(friendly_x.mean()), int(friendly_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [[0], target])

    def run_loop(self, env, max_frames=0, max_episodes=10000, save_checkpoints=500, evaluate_checkpoints=10):
        """A run loop to have agents and an environment interact."""
        total_frames = 0
        start_time = time.time()

        action_spec = env.action_spec()
        observation_spec = env.observation_spec()

        self.setup(observation_spec, action_spec)
        try:
            while self.n_episodes < max_episodes:

                obs = env.reset()[0]
                # remove unit selection from the equation by selecting the entire army on every new game.
                select_army = actions.FunctionCall(_SELECT_ARMY, [[False]])
                obs = env.step([select_army])[0]

                self.reset()
                episode_reward = 0

                while True:
                    total_frames += 1

                    self.obs = obs.observation["feature_screen"][self.features]
                    s = np.expand_dims(self.obs, 0)

                    if max_frames and total_frames >= max_frames:
                        print("max frames reached")
                        return
                    if obs.last():
                        print(f"Episode {self.n_episodes + 1}:\t total frames: {total_frames} Epsilon: {self._epsilon.value()}")
                        self._epsilon.increment()
                        break

                    action = self.get_action(s)
                    env_actions = self.get_env_action(action, obs, command=_ATTACK_SCREEN)
                    obs = env.step([env_actions])[0]
                    r = obs.reward
                    episode_reward += r
                    s1 = np.expand_dims(obs.observation["feature_screen"][self.features], 0)
                    done = r > 0
                    if self._epsilon.isTraining:
                        transition = Transition(s, action, s1, r, done)
                        self._memory.push(transition)

                    if total_frames % self.train_q_per_step == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self.train_q()

                    if total_frames % self.target_q_update_frequency == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self._Qt = copy.deepcopy(self._Q)

                if evaluate_checkpoints > 0 and ((self.n_episodes % evaluate_checkpoints) - (evaluate_checkpoints - 1) == 0 or self.n_episodes == 0):
                    print('Evaluating...')
                    self._epsilon.isTraining = False  # we need to make sure that we act greedily when we evaluate
                    self.run_loop(env, max_episodes=max_episodes, evaluate_checkpoints=0)
                    self._epsilon.isTraining = True
                if evaluate_checkpoints == 0:  # this should only activate when we're inside the evaluation loop
                    self.reward.append(episode_reward)
                    print(f'Evaluation Complete: Episode reward = {episode_reward}')
                    break

                self.n_episodes += 1
                if len(self._loss) > 0:
                    self.loss.append(self._loss[-1])
                    self.max_q.append(self._max_q[-1])
                if self.n_episodes % save_checkpoints == 0:
                    if self.n_episodes > 0:
                        self.save_data(episodes_done=self.n_episodes)

        except KeyboardInterrupt:
            pass
        finally:
            print("finished")
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


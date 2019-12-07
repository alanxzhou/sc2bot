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
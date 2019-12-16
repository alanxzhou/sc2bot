import pickle
import numpy as np
import time
from pysc2 import maps
from pysc2.agents.scripted_agent import _xy_locs
from pysc2.agents.base_agent import BaseAgent
from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features
from pysc2.maps import lib
from sc2bot.utils.replay_memory import ReplayMemory, Transition
from sc2bot.utils import custom_maps

from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_string("map1", "DefeatRoachesAntiSuicide", "Name of a map to use.")
flags.DEFINE_string("map2", "DefeatRoachesAntiSuicideMarineDeath0", "Name of a map to use.")
flags.mark_flag_as_required("map1")
flags.mark_flag_as_required("map2")

_UNIT_TYPE = 6
_SELECTED = 7
_UNIT_HIT_POINTS = 8
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

FUNCTIONS = actions.FUNCTIONS
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals


class BattleAgentScriptedBeacon(BaseAgent):

    def __init__(self, mapname):
        super(BattleAgentScriptedBeacon, self).__init__()
        self.max_frames = int(1e5)
        self._memory = ReplayMemory(self.max_frames)
        self.obs = None
        self.features = 5
        self.mapname = mapname
        self.screen_size = 32

    def get_action(self, obs):
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = np.mean(beacon, axis=0).round().astype(int)
            return FUNCTIONS.Move_screen("now", beacon_center)
        else:
            return FUNCTIONS.select_army("select")

    def run_loop(self, env):
        """A run loop to have agents and an environment interact."""
        start_time = time.time()
        total_frames = 0

        action_spec = env.action_spec()
        observation_spec = env.observation_spec()

        self.setup(observation_spec, action_spec)
        try:
            while True:

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

                    if total_frames >= self.max_frames:
                        pickle.dump(self._memory.memory, open(f'./data/{self.mapname}/scripted_replaymemory_res{self.screen_size}.pkl', 'wb'))
                        print("max frames reached")
                        return
                    if obs.last():
                        break

                    action = self.get_action(obs)
                    if len(action[-1]) > 0:
                        a_idx = action[-1][-1]
                        action_indices = [a_idx[1], a_idx[0]]
                        action_index = np.ravel_multi_index(action_indices, [self.screen_size, self.screen_size])

                    obs = env.step([action])[0]

                    r = obs.reward
                    episode_reward += r
                    s1 = np.expand_dims(obs.observation["feature_screen"][self.features], 0)
                    done = r > 0
                    if len(action[-1]) > 0:
                        transition = Transition(s, action_index, s1, r, done)
                        self._memory.push(transition)

                print(f'Total frames: {total_frames}')
        finally:
            print("finished")
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


def run_thread(map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.protoss,
                                 sc2_env.Difficulty.very_easy)],
            step_mul=8,
            game_steps_per_episode=0,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=32,
                                                       minimap=32)),
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = BattleAgentScriptedBeacon(map_name)
        agent.run_loop(env)


def main(unused_argument):
    map_names = ['MoveToBeacon']
    for map_name in map_names:
        get = lib.get(map_name)
        maps.get(map_name)
        run_thread(map_name, FLAGS.render)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == '__main__':
    app.run(main)

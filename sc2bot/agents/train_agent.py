from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from absl import app
from absl import flags

# from sc2bot.agents.rl_agent import BaseRLAgent as Agent
from sc2bot.agents.battle_agent import *
from sc2bot.agents.beacon_agent import BeaconAgent
from sc2bot.utils import custom_maps
from pysc2.maps import lib

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("train", True, "Whether we are training or running")
flags.DEFINE_integer("screen_resolution", 32,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "BeaconAgent", "Which agent to run")
flags.DEFINE_enum("agent_race", None, [str(i) for i in list(sc2_env.Race)], "Agent's race.")
flags.DEFINE_enum("bot_race", None, [str(i) for i in list(sc2_env.Race)], "Bot's race.")
flags.DEFINE_enum("difficulty", None, [str(i) for i in list(sc2_env.Difficulty)],
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("load_checkpoint", False, "Whether or not to load checkpoint from previous training session")
flags.DEFINE_bool("load_params", False, "Whether or not to load parameters from previous training session")
flags.DEFINE_string("load_file", f'./data/MoveToBeacon/beacon_13149steps_32dim', "file to load params from")
flags.DEFINE_string("save_file", '', "file to save params to / load from if not loading from checkpoint")

flags.DEFINE_integer("max_episodes", 10000, "Maximum number of episodes to train on")
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
# flags.DEFINE_string("map", "DefeatRoachesAntiSuicideMarineDeath0", "Name of a map to use.")
# flags.DEFINE_string("map", "DefeatZerglingsAndBanelings", "Name of a map to use")
flags.mark_flag_as_required("map")


def run_thread(map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.protoss,
                                 sc2_env.Difficulty.very_easy)],
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=FLAGS.screen_resolution,
                                                       minimap=FLAGS.minimap_resolution)),
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        if not FLAGS.save_file:
            save_name = f'./data/{FLAGS.map}/{FLAGS.max_episodes}eps_{FLAGS.agent}'
        else:
            save_name = FLAGS.save_file
        if FLAGS.load_checkpoint:
            agent = agent_selector(FLAGS.agent, save_name=save_name, load_name=FLAGS.load_file)
            # agent = Agent(save_name=FLAGS.load_file)
            agent.load_model_checkpoint(load_params=FLAGS.load_params)
        else:
            # save_name = f'./data/{FLAGS.map}/{FLAGS.max_episodes}eps_{FLAGS.agent}'
            agent = agent_selector(FLAGS.agent, save_name=save_name)
            # agent = Agent(save_name=save_name)
        if FLAGS.train:
            agent.train(env, FLAGS.train,  max_episodes=FLAGS.max_episodes)
        else:
            agent.evaluate(env)


def agent_selector(agent_string, save_name=None, load_name=None):
    if agent_string == 'BattleAgentLimited':
        return BattleAgentLimited(save_name=save_name, load_name=load_name)
    elif agent_string == 'BattleAgent':
        return BattleAgent(save_name=save_name, load_name=load_name)
    elif agent_string == 'BeaconAgent':
        return BeaconAgent(save_name=save_name, load_name=load_name)
    elif agent_string == 'BattleAgentBeacon':
        return BattleAgentBeacon(save_name=save_name, load_name=load_name)


def main(unused_argv):
    """Run an agent."""
    # stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    # stopwatch.sw.trace = FLAGS.trace

    get = lib.get(FLAGS.map)
    maps.get(FLAGS.map)  # Assert the map exists.
    run_thread(FLAGS.map, FLAGS.render)

    # if FLAGS.profile:
    #   print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)

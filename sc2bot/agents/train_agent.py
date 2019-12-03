from pysc2 import maps
from pysc2.env import available_actions_printer
# from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import features

from absl import app
from absl import flags

from sc2bot.agents.rl_agent import BaseRLAgent as Agent

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("train", True, "Whether we are training or running")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32,
                     "Resolution for minimap feature layers.")

# flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent", "Which agent to run")
flags.DEFINE_enum("agent_race", None, [str(i) for i in list(sc2_env.Race)], "Agent's race.")
flags.DEFINE_enum("bot_race", None, [str(i) for i in list(sc2_env.Race)], "Bot's race.")
flags.DEFINE_enum("difficulty", None, [str(i) for i in list(sc2_env.Difficulty)],
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_integer("max_episodes", 5000, "Maximum number of episodes to train on")
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
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
        agent = Agent()
        # run_loop([agent], env, FLAGS.max_agent_steps)
        # agent.train(env, FLAGS.train)
        agent.train(env, FLAGS.train,  max_episodes=FLAGS.max_episodes,
                    save_name=f'./data/{FLAGS.map}_{FLAGS.max_episodes}_{FLAGS.screen_resolution}{FLAGS.minimap_resolution}')
        if FLAGS.save_replay:
            env.save_replay(Agent.__name__)


def main(unused_argv):
    """Run an agent."""
    # stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    # stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.
    run_thread(FLAGS.map, FLAGS.render)

    # if FLAGS.profile:
    #   print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)

#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import importlib
import threading
import time
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
# from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch, features

from absl import app
from absl import flags

from agents.beacon_agent import BeaconAgent as Agent


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run")
flags.DEFINE_enum("agent_race", None, [str(i) for i in list(sc2_env.Race)], "Agent's race.")
flags.DEFINE_enum("bot_race", None, [str(i) for i in list(sc2_env.Race)], "Bot's race.")
flags.DEFINE_enum("difficulty", None, [str(i) for i in list(sc2_env.Difficulty)],
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.mark_flag_as_required("map")


def run_thread(agent_cls, map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.protoss,
                                 sc2_env.Difficulty.very_easy)],
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=32)),
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = agent_cls()
        run_loop([agent], env, FLAGS.max_agent_steps)
        if FLAGS.save_replay:
            env.save_replay(agent_cls.__name__)


def run_loop(agents, env, max_frames=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    start_time = time.time()

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    for agent in agents:
        agent.setup(observation_spec, action_spec)

    try:
        while True:
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                total_frames += 1
                actions = [agent.step(timestep)
                           for agent, timestep in zip(agents, timesteps)]
                if max_frames and total_frames >= max_frames:
                    return
                if timesteps[0].last():
                    break
                timesteps = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))


def main(unused_argv):
    """Run an agent."""
    # stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    # stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.

    # agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    # agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_cls = Agent

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread, args=(agent_cls, FLAGS.map, False))
        threads.append(t)
        t.start()

    run_thread(agent_cls, FLAGS.map, FLAGS.render)

    for t in threads:
        t.join()

    # if FLAGS.profile:
        # print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
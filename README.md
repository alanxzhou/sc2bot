# sc2bot

sc2bot is a python based package meant to train a deep-learning based reinforcement agent on the Move To Beacon and Defeat Roaches mini games distributed by Google DeepMind.

# Quick Start
## Installation
sc2bot was created and tested in Python 3.7+ and was tested on macOS, Linux, and Windows, but was primarily developed in Windows 10. It requires the following packages:
* pysc2
* numpy
* matplotlib
* pytorch 
sc2bot also requires an installation of StarCraft II somewhere on your machine (discussed later)

Clone the reposistory by running the following in your terminal:
```
git clone https://github.com/alanxzhou/sc2bot
```
Then navigate to the sc2bot folder and run
```
python setup.py install
```
This should install the package and all other requirements.

## StarCraft II
The following instructions are largely copied from [PySC2's instructions for installation](https://github.com/deepmind/pysc2)

sc2bot requires a full installation of StarCraft II and only works on versions that include the API, which is 3.16.1 and above.

### Linux
Follow Blizzard's documentation to get the linux version. By default, sc2bot expects the game to live in \~/StarCraftII/. You can override this path by setting the SC2PATH environment variable or creating your own run_config.

### Windows / MacOS
Install of the game as normal from [Battle.net](https://battle.net). Even the [Starter Edition](https://starcraft2.com/en-us/) will work. If you used the default install location PySC2 should find the latest binary. If you changed the install location, you might need to set the SC2PATH environment variable with the correct location in Environment Variables. If the variable doesn't exist you will need to create it as a new variable.

## Maps
In order to use a map, it must exist in the StarCraftII `Maps` directory before they can be played. To use the default maps used by PySC2, download them at [Blizzard's API repository](https://github.com/Blizzard/s2client-proto#downloads) and extract them to the `StarCraftII/Maps/` directory (you'll need to create the directory if it doesn't exist yet). 

There are also custom maps made by me in this repository in the `maps` folder. The map labeled `DefeatRoachesAntiSuicideMarineDeath0.SC2Map` is the map used for my experiments in the report. In order to use these maps, you need to put them in `StarCraftII/Maps/custom`. If you want to make your own custom maps I suggest you add them to that folder. You'll also need to add them to `sc2bot/utils/custom_maps.py` so that the package can recognize them. Look into PySC2's documentation for more details on custom maps.

## Training an agent
To train an agent with the default parameters, navigate to `sc2bot/agents` and run
```
python train_agent.py
```
This should train an agent on the beacon task. If you want to train with different options, there are many flags that you can use. For example, if you want to train the BattleAgent on the roach task, run 
```
python train_agent.py --agent=BattleAgentBeacon --map=DefeatRoachesAntiSuicideMarineDeath0'
```
This should train the BattleAgent on `DefeatRoachesAntiSuicideMarineDeath0.SC2Map` if you correctly followed the above instructions to import maps.
To evaluate an agent, run
```
python train_agent.py --train=False
```
There are a lot of flags that come with `train_agent.py`. Look in the source code for more detailed descriptions but the following is a quick rundown of the most useful ones not already mentioned:

* render (bool): whether or not to render the PySC2 interface while rendering the game
* screen_resolution (int): the spacing of the discretization of the game screen
* load_checkpoint (bool): Whether or not to load weights from a previous training session
* load_params (bool): Whether or not to load parameters from a previous training session
* load_file (str): the name of the file from which to load weights and / or parameters from a previous training session
* max_episodes (int): the number of episodes to train on (note that turning the train flag off means the agent will run indefinitely)

The load_checkpoint, load_params, and load_file flags are a little tricky to use but it goes something like this: load_checkpoint only specifies if you want to load weights for the DQN from a `.pth` file. load_params specifies whether if you want to load the measurements as well as the epsilon value from a previous run. load_file is the name of the file without the file type name (e.g., `.pth`), and in the case of load_params, without the footer `_data.pkl.` When training, parameters and weights should both be saved under the same filename, but with these two separate footers. fFor the load_file flag, just type in the filename without the footers. Generally, loading the parameters won't be necessary unless you plan on doing experiments yourself. As an example of loading parameters, this package has weights for the BeaconAgent included. To see this agent in action run
```
python train_agent.py --train=False --agent=BeaconAgent --map=MoveToBeacon --load_checkpoint=True --load_params=False --load_file='./data/MoveToBeacon/beacon_13149steps_32dim'
```

### Pretraining an agent with experience replay from a scripted agent
To generate experience replay from a scripted agent navigate to the `sc2bot/agents` folder and run (for the BeaconAgent)
```
python scripted_agent_beacon.py
```
and then
```
python pretrain_beacon_agent.py
```
These two scripts in conjunction will create an experience replay from the scripted agents and then pretrain the DQNs on this experience. You may need to change some path references in order to get this to work. If you want to pretrain on the Roaches agent you'll need to replace the above scripts with `scripted_agent.py` and `pretrain_agent.py` respectively.


# Acknowledgements
Thanks to Jivko and Gyan for a great semester in Comp150-06.
Thanks to Steveb Brown's tutorials on using PySC2 for allowing to run my code at all: ([https://github.com/skjb](https://github.com/skjb))
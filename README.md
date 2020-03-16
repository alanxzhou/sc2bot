# sc2bot

sc2bot is a python based package meant to train a deep-learning based reinforcement agent on the Move To Beacon and Defeat Roaches mini games distributed by deepmind.

# Quick Start
## Installation
sc2bot was created and tested in Python 3.7+ and requires the following packages:
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
Install the game as normal from [Battle.net](https://battle.net). Even the [Starter Edition](https://starcraft2.com/en-us/) will work. If you used the default install location PySC2 should find the latest binary. If you changed the install location, you might need to set the SC2PATH environment variable with the correct location in Environment Variables. If the variable doesn't exist you will need to create it as a new variable.

## Maps
In order to use a map, it must exist in the StarCraftII `Maps` directory before they can be played. To use the default maps used by PySC2, download them at [Blizzard's API repository](https://github.com/Blizzard/s2client-proto#downloads) and extract them to the `StarCraftII/Maps/` directory (you'll need to create the directory if it doesn't exist yet). 

# Acknowledgements
Thanks to Jivko and Gyan for a great semester and teaching me RL.
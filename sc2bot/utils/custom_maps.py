from pysc2.maps import lib


class CustomGame(lib.Map):
    directory = "custom"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


mini_games = [
    "DefeatRoachesAntiSuicide",  # 120s
    "DefeatRoachesAntiSuicideMarineDeath0"
]


for name in mini_games:
    globals()[name] = type(name, (CustomGame,), dict(filename=name))

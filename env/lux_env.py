
from lux.game import Game
from kaggle_environments import make
from typing import Dict
from opponent.lux_ai.rl_agent.rl_agent import agent

DEBUG = True
LOG_LEVEL = 1

class ProxyEnv():
    def __init__(self,map_size=None,seed=None):
        self.conf = {"loglevel": LOG_LEVEL}
        if map_size is not None:
            self.conf["height"] = map_size
            self.conf["width"] = map_size
        if seed is not None:
            self.conf["seed"] = seed


    def reset(self):
        obs = {}
        self.env = make("lux_ai_2021",
                        configuration=self.conf, 
                        debug=DEBUG)
        self.game_state = Game()
        infos = self.env.reset(2)
        obs[0] = infos[0]["observation"]
        obs[1] = infos[1]["observation"]
        self.game_state._initialize(obs[0]["updates"])
        self.game_state._update(obs[0]["updates"][2:])
        self.game_state.id = obs[0]['player']
        return obs, self.game_state, self.env.done

    def step(self, actions):
        obs = {}
        infos = self.env.step(actions)
        obs[0] = infos[0]["observation"]
        obs[1] = infos[1]["observation"]
        # print(obs[0]['updates'])
        self.game_state._update(obs[0]["updates"])
        return obs, self.game_state, self.env.done

    def save(self, path='replay.json'):
        import json
        with open(path, 'w') as f:
            json.dump(self.env.toJSON(), f)


class Observation(Dict[str, any]):
    def __init__(self, player=0):
        self.player = player
        # self.updates = []
        # self.step = 0
    



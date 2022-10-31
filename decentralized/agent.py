import sys,os
sys.path.append((os.path.dirname(os.path.abspath(__file__))))
from lux.game import Game
import decentralized.deAgent as deAgent

game_state = None
plf_agent_old = deAgent.Agent()


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    ### AI Code goes down here! ### 
    player_id = observation.player
    actions = plf_agent_old.get_actions(game_state,observation,player_id)
    return actions

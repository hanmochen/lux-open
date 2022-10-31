import numpy as np
from copy import deepcopy

# UNIT_INCRE_WEIGHT = 0.01
# RP_INCRE_WEIGHT = 0.001
# ACHIEVE_COAL_REWARD = 0.05
# ACHIEVE_URN_REWARD = 0.2
# CAN_COAL_THRESHOLD = 50
# CAN_URN_THRESHOLD = 200

# CT_INCRE_WEIGHT = 1
# FUEL_INCRE_WEIGHT = 0.005
# GAME_RESULT = 10
# STEP = 0.01


class RewardParser(object):
    def __init__(self, reward_param):

        self.reward_param = reward_param
    
    def reset(self):
        pass

    def process(self, done, global_info):

        reward = [0, 0]
        if done:
            final_result_0 = 10000.0 * global_info["ct_count"][0] + global_info["unit_count"][0]
            final_result_1 = 10000.0 * global_info["ct_count"][1] + global_info["unit_count"][1]
            if final_result_0 > final_result_1:
                reward = [1., -1]
            elif final_result_0 < final_result_1:
                reward = [-1, 1.]
            print('********',done, reward,global_info['ct_count'])

        return reward, [[0,0,0,0,0], [0,0,0,0,0]]


        
        
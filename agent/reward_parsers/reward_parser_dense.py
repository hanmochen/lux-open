import numpy as np
from copy import deepcopy

# UNIT_INCRE_WEIGHT = 0.01
# CT_INCRE_WEIGHT = 0.01
# RP_INCRE_WEIGHT = 0.001
# ACHIEVE_COAL_REWARD = 0.05
# ACHIEVE_URN_REWARD = 0.2
# CAN_COAL_THRESHOLD = 50
# CAN_URN_THRESHOLD = 200

class RewardParser(object):
    def __init__(self, reward_param):
        super(RewardParser, self).__init__()
        self.is_init = False
        self.last_rp = None
        self.last_ct_count = None
        self.last_unit_count = None
        self.reward_param = reward_param

    def reset(self):
        self.is_init = False
        self.last_rp = None
        self.last_ct_count = None
        self.last_unit_count = None
        self.last_total_fuel = None

    def process(self, done, global_info):
        sub_rewards = [[0,0,0,0,0], [0,0,0,0,0]]
        if self.is_init:
            for team in [0,1]:
                unit_increment = global_info["unit_count"][team] - self.last_unit_count[team]
                ct_increment = global_info["ct_count"][team] - self.last_ct_count[team]
                rp_increment = global_info["rp"][team] - self.last_rp[team]
                fuel_increment = max(0, global_info["total_fuel"][team] - self.last_total_fuel[team])
                sub_rewards[team][0] = ct_increment * self.reward_param["ct_reward_weight"]
                sub_rewards[team][1] = unit_increment * self.reward_param["unit_reward_weight"]
                sub_rewards[team][2] = rp_increment * self.reward_param["rp_reward_weight"]
                sub_rewards[team][3] = fuel_increment * self.reward_param["fuel_reward_weight"]

                if done:
                    sub_rewards[team][4] = self.reward_param["win_reward"] * np.sqrt(max(0,global_info["ct_count"][team]-global_info["ct_count"][1-team]))
                
        rewards = [sum(sub_rewards[0]), sum(sub_rewards[1])]

        self.last_rp = deepcopy(global_info["rp"])
        self.last_ct_count = deepcopy(global_info["ct_count"])
        self.last_total_fuel = deepcopy(global_info["total_fuel"])
        self.last_unit_count = deepcopy(global_info["unit_count"])
        self.is_init = True
                   
        return rewards, sub_rewards
        
        
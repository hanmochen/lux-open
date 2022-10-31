
import numpy as np
import torch
import random
import torch.nn.functional as F
from env.actions import Worker, Cart, Citytile
from decentralized.feature_parser import DeFeatureParser
from decentralized.deNet import DeNet


# Set Model Path
import os

current_path = os.path.dirname(__file__)

mapsizes = [12,16,24,32]
model_paths = {}
for mapsize in [12,16,24,32]:
    model_paths[mapsize] = os.path.join(current_path, 'models/model_' + str(mapsize) + '.pt')


#Set Log Path
#import logging
#log_path = os.path.join(current_path, 'log.txt')
#logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode="a+",
#                         format="%(asctime)-15s %(levelname)-8s %(message)s")
#logging.info("hello")

is_hard_choice = False 
localmap_size = 15
is_trans = False

train_map_size = 16 # 12 32 24 16

worker_act_dim, city_act_dim = [7, 3] # worker move  e w n s c / build / none  city build worker / research / none -- we distinguish worker_stay and none

act_dims = [worker_act_dim, city_act_dim] # self.cart_act_dim, self.worker_act_dim, self.city_act_dim


global_dim = 698
self_dim = 405 # global self
imagelike_dim = [17, localmap_size, localmap_size] # [17,15,15] / [17,31,31]

unit_single_dim = 9
city_single_dim = 9 
unit_num = 160
city_num = 160

unit_dim = unit_single_dim*unit_num
city_dim = city_single_dim*city_num

nearest_num = 20

feature_dims = [global_dim, self_dim, unit_dim, city_dim, imagelike_dim, unit_single_dim, city_single_dim, unit_num, city_num, nearest_num]

class Agent():
    def __init__(self):
        self.worker_act_dim, self.city_act_dim = act_dims
        self.unit_num = unit_num
        self.city_num = city_num
        self.unit_single_dim = unit_single_dim
        self.city_single_dim = city_single_dim
        self.is_hard_choice = is_hard_choice

        self.maxsize = 32
        self.split_dim = [
            global_dim,
            self_dim, 
            unit_dim, 
            city_dim, 
            unit_dim, 
            city_dim, 
            np.prod(imagelike_dim)
        ]

        self.worker = Worker()
        self.cart = Cart()
        self.citytile = Citytile()


        self.ft_parser = DeFeatureParser(15,is_trans)
        self.policies = {}
        for mapsize in mapsizes:
            policy = DeNet(feature_dims, act_dims)
            policy.load_state_dict(torch.load(model_paths[mapsize], map_location=torch.device('cpu')))
            self.policies[mapsize] = policy
        self.id_info = None
        self.global_info = None

    def parse_obs(self, game_state, observation, team_id):

        obs_list_n, self.id_info, self.map_info, self.global_info = self.ft_parser.parse(observation, game_state, team_id)
        
        self.mapsize = self.global_info['map_size'][0]

        return obs_list_n

    def get_actions(self, game_state, observation, team_id):


        def _choice(pi):
            if self.is_hard_choice: return int(np.argmax(pi))
            sum_pi = sum(pi)
            rand = random.random()*sum_pi
            sum_prob = 0.0
            for i, prob in enumerate(pi):
                sum_prob += prob
                if rand <= sum_prob:
                    return i
            return len(pi) - 1
        
        #logging.info(observation)
        idx_actions = {}
        obs_list_n = self.parse_obs(game_state, observation, team_id)

        
        city_left = self.global_info["ct_count"][team_id]
        unit_cap = max(city_left - self.global_info["unit_count"][team_id], 0)
        research_point = self.global_info["rp"][team_id]
        # logging.info("research_point: {}".format(research_point))

        for idx in range(len(obs_list_n)):
            
            if (self.id_info[idx]["team"] == team_id) and (self.id_info[idx]["is_alive"]) and (self.id_info[idx]["cooldown"] < 0.999):

                if self.id_info[idx]["type"] == "worker":

                    with torch.no_grad():
                        _feature = torch.split(torch.unsqueeze(torch.tensor(obs_list_n[idx]), dim=0), self.split_dim, dim=1)
                        softmax_logits = self.net_infer(_feature)[0]
                        valid_worker = self.get_valid_actions(idx,team_id)
                        worker_action_pi = np.multiply(softmax_logits, valid_worker)
                        worker_action_target = _choice(worker_action_pi)
                        
                    idx_actions[idx] = worker_action_target
                
                elif self.id_info[idx]["type"] == "citytile":

                    if unit_cap > 0 :
                        p = min(1,2*unit_cap/city_left)
                        city_action = np.random.choice([0, 1], p = [p,1-p])
                        if city_action == 0:
                            unit_cap -= 1
                    else: 
                        city_action = 1

                    if (city_action == 1) and (research_point >= 200): 
                        city_action = 2
                    
                    city_left = max(1, city_left - 1)
                    if city_action == 1:
                        research_point += 1
                    idx_actions[idx] = city_action


        actions = self.parse_actions(idx_actions,team_id)

        return actions
    
    def net_infer(self, _feature):
        imagelike_feature = _feature[-1]
        global_feature, self_feature, unit_feature, city_feature, opunit_feature, opcity_feature = [*_feature[:-1]]
        unit_feature = unit_feature.reshape([self.unit_num, self.unit_single_dim]).unsqueeze(0)
        city_feature = city_feature.reshape([self.city_num, self.city_single_dim]).unsqueeze(0)
        opunit_feature = opunit_feature.reshape([self.unit_num, self.unit_single_dim]).unsqueeze(0)
        opcity_feature = opcity_feature.reshape([self.city_num, self.city_single_dim]).unsqueeze(0)
        imagelike_feature = torch.unsqueeze(imagelike_feature.reshape(*imagelike_dim), dim=0)

        x = (global_feature, self_feature, unit_feature, city_feature, opunit_feature, opcity_feature, imagelike_feature)

        # logging.info("Using Policy {} Loading from ckpt {}".format(self.mapsize,model_paths[self.mapsize]))

        logits = self.policies[self.mapsize].forward(x)
        # logging.info("logits: {}".format(logits))

        softmax_logits = F.softmax(logits, dim=1)
        # logging.info("softmax_logits: {}".format(softmax_logits))

        return softmax_logits

    def get_valid_actions(self, idx, team_id):
        
        move_directions = [[1,0], [-1,0], [0,-1], [0,1]] # 0: "e", 1: "w", 2: "n", 3: "s"
        valid_worker = np.zeros(self.worker_act_dim, dtype=np.float32)

        shift_left = round((self.maxsize - self.mapsize)/2)
        shift_right = round(self.maxsize - (self.maxsize - self.mapsize)/2)

        idx_team = team_id
        op_team = 1 - team_id

        self.can_move_map = np.zeros([self.maxsize+1,self.maxsize+1])
        for pos_x in range(self.maxsize):
            for pos_y in range(self.maxsize):
                if shift_left <= pos_x < shift_right and shift_left <= pos_y < shift_right:
                    if self.map_info["ct_"+str(idx_team)][pos_x, pos_y] == 1:
                        self.can_move_map[pos_x, pos_y] = 1
                    elif self.map_info["ct_"+str(op_team)][pos_x, pos_y] == 1 or self.map_info["uids"][pos_x, pos_y]:
                        self.can_move_map[pos_x, pos_y] = 0
                    else:
                        self.can_move_map[pos_x][pos_y] = 1

        worker_pos = self.id_info[idx]["loc"][idx_team]
        worker_resource = self.id_info[idx]["wood_carry"] + self.id_info[idx]["coal_carry"] + self.id_info[idx]["uranium_carry"]

        if worker_resource >= 99.9 and self.map_info["ct_"+str(idx_team)][worker_pos[0], worker_pos[1]] != 1 and self.map_info["is_res"][worker_pos[0], worker_pos[1]] != 1:
            valid_worker[5] = 1.

        # move is not valid (1) opponent team's city (2) other units (3) out of the map
        for i, direction in enumerate(move_directions):
            target_pos = [worker_pos[0] + direction[0], worker_pos[1] + direction[1]]
            valid_worker[i] = self.can_move_map[target_pos[0],target_pos[1]]

        if sum(valid_worker) == 0:
            valid_worker[-1] = 1.

        return valid_worker
   
    
    def parse_actions(self, actions, team_id):
        # in feature parser: we shift the map first and then 
        # worker citytile
        move_directions = [[1,0], [-1,0], [0,-1], [0,1]] # 0: "e", 1: "w", 2: "n", 3: "s"
        move_mapping = {0: "e", 1: "w", 2: "n", 3: "s", 4: "c"} # [[1,0], [-1,0], [0,-1], [0,1]] # 0: "e", 1: "w", 2: "n", 3: "s", 4: "c"
        shift_size = round((self.maxsize - self.mapsize)/2)
        decisions = []
    
        for idx in actions:
            if self.id_info[idx]["type"] == "citytile": 
                city_action = actions[idx]
                city_x, city_y = self.id_info[idx]["loc"][team_id]
                if city_action == 0:
                    decision = self.citytile.build_worker(city_x - shift_size, city_y - shift_size)
                elif city_action == 1:
                    decision = self.citytile.research(city_x - shift_size, city_y - shift_size)
                else:
                    decision = []


            elif self.id_info[idx]["type"] == "worker":

                worker_action = actions[idx]
                
                if 0 <= worker_action <= 3:
                    move_str = move_mapping[worker_action]
                    worker_pos = self.id_info[idx]["loc"][team_id]
                    direction = move_directions[worker_action]
                    target_pos = [worker_pos[0] + direction[0], worker_pos[1] + direction[1]]
                    if self.can_move_map[target_pos[0], target_pos[1]] == 1:
                        decision = self.worker.move(self.id_info[idx]["uid"], move_str)
                        if self.map_info["ct_"+str(team_id)][target_pos[0], target_pos[1]] == 0:
                            self.can_move_map[target_pos[0], target_pos[1]] = 0
                        self.can_move_map[worker_pos[0], worker_pos[1]] = 1
                    else:
                        decision = []
                elif worker_action == 5:
                    decision = self.worker.build_city(self.id_info[idx]["uid"])
                else:
                    decision = []

            if decision:
                decisions.append(decision)  

        return decisions

    

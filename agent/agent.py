
from agent.feature_parser import FeatureParser
from agent.net import Net
from env.actions import Worker, Citytile, Cart
import torch
import numpy as np
from functools import partial
import copy

if_city_rule = False

np.set_printoptions(threshold=np.inf)
unit_action_mapping = {0: "worker", 1: "cart", 2: "ct"}
directions_mapping = {"e": [1,0], "w": [-1,0], "n": [0,-1], "s": [0,1],
                       0 : [1,0],  1 : [-1,0],  2 : [0,-1],  3 : [0,1]}
soft_or_hard_max = "soft"
resolve_collision = True

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def choose_action(valid_actions,logits,soft_or_hard_max=soft_or_hard_max):
    _logits = _softmax(logits[valid_actions!= 0])
    if soft_or_hard_max == 'soft':
        action = np.random.choice(valid_actions.nonzero()[0], p=_logits)
    else: 
        action = valid_actions.nonzero()[0][np.argmax(_logits)]
    return action

map_size = 12
model_param={
    "use_transformer": False,
    "emb_dim": 9,
    "global_channel": 32,
    "all_channel": 128,
    "n_res_blocks": 8,
}

model_path = None

config = {
    'map_size': map_size,
    'num_day': 9,
    'num_hour': 40,
    'num_hour_daytime': 30,
    'wood_regrow_ub': 500,
    'cooldown_action_ub': 1,
    'resource_ub_worker': 100,
    'resource_ub_cart': 2000,
    'cost_base_city': 23,
    'cost_save_city': 5,
    'research_point_coal': 50,
    'research_point_urn': 200,
    'dict_fuel_value': {'wood': 1, 'coal': 10, 'uranium': 40},
    'cost_worker_city': 0,
    'cost_cart_city': 0,
    'cost_worker_out': 4,
    'cost_cart_out': 10,
    "global_feature_dims": [51, 18],
    "map_channel": 37,
    "model_path": model_path,
    "n_actions": [19,17,4],
    "model_param": model_param,
}

class Agent():
    def __init__(self, config=config):
        self.config = config
        self.feature_parser = FeatureParser(config)
        self.net = Net(config['model_param'], config['global_feature_dims'], config['map_channel'], config['map_size'], config['n_actions'])
        self.net.load_state_dict(torch.load(config["model_path"],map_location=torch.device('cpu')))
        self.map_size = config['map_size']
        self.worker_act_dim, self.cart_act_dim, self.ct_act_dim = config['n_actions']
        self.worker = Worker()
        self.citytile = Citytile()
        self.cart = Cart()
        
    def get_actions(self,obs,conf):
        player_id = obs.player
        # print("func - get_actions - execute - player_id is {}".format(player_id))
        obs_list, self.unit_info,self.map_info,self.global_info = self.feature_parser.parse(obs)
        # logging.info(self.global_info['ct_count'])
        global_emb_feature, global_no_emb_feature, map_feature = np.split(obs_list[player_id], [self.config['global_feature_dims'][0], self.config['global_feature_dims'][0] + self.config['global_feature_dims'][1]])
        global_emb_feature = torch.from_numpy(global_emb_feature).float().unsqueeze(0)
        global_no_emb_feature = torch.from_numpy(global_no_emb_feature).float().unsqueeze(0)
        map_feature = torch.from_numpy(map_feature).float().reshape(1, self.config['map_channel'], self.config['map_size'], self.config['map_size'])
        logits, _ = self.net((global_emb_feature,global_no_emb_feature,map_feature))
        # logits = logits.squeeze().detach().numpy()
        actions = self._parse_action(logits,team=player_id)
        return actions
    
    def _parse_action(self, action_logits, team):

        # print("len of action_logits is {}".format(len(action_logits)))
        # print("shape of action_logits[2] citytile is {}".format(action_logits[2].shape))

        valid_actions = self.get_valid_actions(team)

        # a static valid info 
        self.ct_count = self.global_info["ct_count"][team]
            
        # three dynamic valid info
        self.unit_count = self.global_info["unit_count"][team]
        self.rp_count = self.global_info["rp"][team]
        self.can_move_map = np.zeros([self.map_size, self.map_size])

        # for citytile, the order of loc
        logits_ct = action_logits[2][0]
        valid_ct_mask = copy.deepcopy(valid_actions[2])
        valid_ct_mask = torch.tensor(valid_ct_mask)
        # print(valid_ct_mask == 0)
        valid_ct_mask = torch.where((valid_ct_mask == 0), torch.full_like(valid_ct_mask, -np.inf), valid_ct_mask)
        logits_ct = logits_ct * valid_ct_mask
        # print("the shape of logits_ct - {}".format(logits_ct.shape))
        max_logits_ct = logits_ct.max(dim=-1)[0]
        ct_index_order = torch.argsort(max_logits_ct, dim=-1, descending=True)
        # print(ct_index_order)
        # print("------------")

        # for worker/cart, the order of loc
        logits_worker = action_logits[0][0]
        valid_worker_mask = copy.deepcopy(valid_actions[0])
        valid_worker_mask = torch.tensor(valid_worker_mask)
        valid_worker_mask = torch.where((valid_worker_mask == 0), torch.full_like(valid_worker_mask, -np.inf), valid_worker_mask)
        logits_worker = logits_worker * valid_worker_mask

        logits_cart = action_logits[1][0]
        valid_cart_mask = copy.deepcopy(valid_actions[1])
        valid_cart_mask = torch.tensor(valid_cart_mask)
        valid_cart_mask = torch.where((valid_cart_mask == 0), torch.full_like(valid_cart_mask, -np.inf), valid_cart_mask)
        logits_cart = logits_cart * valid_cart_mask

        max_logits_unit = torch.cat([logits_worker.max(dim=-1)[0], logits_cart.max(dim=-1)[0]], dim=-1)
        unit_index_order = torch.argsort(max_logits_unit, dim=-1, descending=True)

        logits_worker = action_logits[0][0]
        logits_cart = action_logits[1][0]
        logits_ct = action_logits[2][0]

        # print("the shape of valid_actions[0] is {}".format(valid_actions[0].shape))
        # actions {0: [worker_action, cart_action, city_action]} {1: [worker_action, cart_action, city_action]}
        worker_action_mapping = {
            0: {"type": "none", "act": lambda id: None},
            1: {"type": "move", "act": partial(self.worker.move, dir="e")},
            2: {"type": "move", "act": partial(self.worker.move, dir="w")},
            3: {"type": "move", "act": partial(self.worker.move, dir="n")},
            4: {"type": "move", "act": partial(self.worker.move, dir="s")},
            5: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "wood"), "e"]},
            6: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "wood"), "w"]},
            7: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "wood"), "n"]},
            8: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "wood"), "s"]},
            9: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "coal"), "e"]},
            10: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "coal"), "w"]},
            11: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "coal"), "n"]},
            12: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "coal"), "s"]},
            13: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "uranium"), "e"]},
            14: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "uranium"), "w"]},
            15: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "uranium"), "n"]},
            16: {"type": "transfer", "act": [partial(self.worker.transfer, resourceType = "uranium"), "s"]},
            17: {"type": "pillage", "act": self.worker.pillage},
            18: {"type": "build", "act": self.worker.build_city}
        } # transfer action needs different args, so we should use the keys to distinguish transfer action
        city_action_mapping = {
            0: lambda pos:None,
            1: lambda pos: self.citytile.build_worker(pos_x = pos[0], pos_y = pos[1]),
            2: lambda pos: self.citytile.build_cart(pos_x=pos[0],pos_y=pos[1]),
            3: lambda pos: self.citytile.research(pos_x=pos[0], pos_y=pos[1])
        }
        cart_action_mapping = {
            0: {"type": "none", "act": lambda id: None},
            1: {"type": "move", "act": partial(self.cart.move, dir="e")},
            2: {"type": "move", "act": partial(self.cart.move, dir="w")},
            3: {"type": "move", "act": partial(self.cart.move, dir="n")},
            4: {"type": "move", "act": partial(self.cart.move, dir="s")},
            5: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "wood"), "e"]},
            6: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "wood"), "w"]},
            7: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "wood"), "n"]},
            8: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "wood"), "s"]},
            9: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "coal"), "e"]},
            10: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "coal"), "w"]},
            11: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "coal"), "n"]},
            12: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "coal"), "s"]},
            13: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "uranium"), "e"]},
            14: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "uranium"), "w"]},
            15: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "uranium"), "n"]},
            16: {"type": "transfer", "act": [partial(self.cart.transfer, resourceType = "uranium"), "s"]},
        } # transfer action needs different args, so we should use the keys to distinguish transfer action
            
        decisions = []

        # citytiles' actions
        for idx in ct_index_order:
            decision = None
            pos = (int(idx) // self.map_size, int(idx) % self.map_size)
            logits = logits_ct[idx].detach().numpy()
            # print("the shape of ct_logits - {}".format(logits.shape))
            if self.map_info["act_ct_"+str(team)][pos]:
                # id = self.map_info["act_ct_"+str(team)][pos]
                valid_action_ct = valid_actions[2][idx]
                # print(valid_action_ct)
                # print("------------------------")
                if self.ct_count <= self.unit_count:
                    valid_action_ct[1] = 0.
                    valid_action_ct[2] = 0.
                if self.rp_count >= 200:
                    valid_action_ct[3] = 0.
                # print(valid_action_ct)
                if sum(valid_action_ct) == 0:
                    valid_action_ct[0] = 1.
                action = choose_action(valid_action_ct, logits)
                # print(action)
                # print("------------------------")
                if action in [1,2]:
                    self.unit_count += 1
                if action == 3:
                    self.rp_count += 1
                action = city_action_mapping[action]
                decision = action(pos)
            if decision:
                decisions.append(decision)
            
        # print(decisions)
            
        # initialize the self.can_move_map
        # if self.can_move_map[pos_x][pos_y] == 0: a unit cannot move to that position
        # otherwise, a unit can move to that position
        self.can_move_map = np.ones([self.map_size, self.map_size])
        for pos_x in range(self.map_size):
            for pos_y in range(self.map_size):
                if self.map_info["city_"+str(team)][pos_x, pos_y] == 1:
                    self.can_move_map[pos_x, pos_y] = 1
                elif self.map_info["city_"+str(1-team)][pos_x, pos_y] == 1:
                    self.can_move_map[pos_x, pos_y] = 0
                elif type(self.map_info["unit_"+str(team)][pos_x][pos_y]) == str:
                    # print("unit at ({},{}) is {}".format(pos_x, pos_y, self.map_info["unit_"+str(team)][pos_x][pos_y]))
                    self.can_move_map[pos_x, pos_y] = 0
                elif type(self.map_info["unit_"+str(1-team)][pos_x][pos_y]) == str:
                    # print("unit at ({},{}) is {}".format(pos_x, pos_y, self.map_info["unit_"+str(1-team)][pos_x][pos_y]))
                    self.can_move_map[pos_x, pos_y] = 0
                else:
                    self.can_move_map[pos_x, pos_y] = 1

            # actions of workers and carts
        for i in unit_index_order:
            decision = None
            unit_type, idx = int(i) // (self.map_size*self.map_size), int(i) % (self.map_size*self.map_size)
            pos = (int(idx) // self.map_size, int(idx) % self.map_size) 
            # print(" i is {}, unit type is {}, idx is {}, pos is {}".format(i, unit_type, idx, pos))
            if unit_action_mapping[unit_type] == "worker" and self.map_info["act_worker_"+str(team)][pos[0]][pos[1]]:
                logits = logits_worker[idx].detach().numpy()
                id = self.map_info["act_worker_"+str(team)][pos[0]][pos[1]]
                assert pos == self.unit_info[id]["loc"]
                valid_action_worker = valid_actions[unit_type][idx]
                # decide whether this worker can move
                for dir in range(4):
                    if valid_action_worker[dir+1] == 1:
                        direction = directions_mapping[dir]
                        target_pos = (pos[0]+direction[0], pos[1]+direction[1])
                        if self.can_move_map[target_pos[0], target_pos[1]] == 1:
                            pass
                        else:
                            action_idx = 1 + dir
                            valid_action_worker[action_idx] = 0.
                action = choose_action(valid_action_worker, logits)
                if action in [1,2,3,4]:
                    move_dir = action-1
                action = worker_action_mapping[action]
                if action["type"] == "transfer":
                    direction = action["act"][1]
                    pos_delta = directions_mapping[direction]
                    transfer_pos = (pos[0]+pos_delta[0], pos[1]+pos_delta[1])
                    if self.map_info['unit_'+str(team)][transfer_pos[0]][transfer_pos[1]]:
                        dest_id = self.map_info['unit_'+str(team)][transfer_pos[0]][transfer_pos[1]]
                        decision = action["act"][0](id=id,dest_id=dest_id,amount=2000)
                elif action["type"] == "move":
                    direction = directions_mapping[move_dir]
                    target_pos = (pos[0]+direction[0], pos[1]+direction[1])
                    self.can_move_map[target_pos[0], target_pos[1]] = 0
                    self.can_move_map[pos[0], pos[1]] = 1
                    decision = action["act"](id=id)
                else:
                    decision = action["act"](id=id)

            elif unit_action_mapping[unit_type] == "cart" and self.map_info["act_cart_"+str(team)][pos[0]][pos[1]]:
                logits = logits_cart[idx].detach().numpy()
                id = self.map_info["act_cart_"+str(team)][pos[0]][pos[1]]
                assert pos == self.unit_info[id]["loc"]
                valid_action_cart = valid_actions[unit_type][idx]
                    # decide whether this cart can move
                for dir in range(4):
                    if valid_action_cart[dir+1] == 1:
                        direction = directions_mapping[dir]
                        target_pos = (pos[0]+direction[0], pos[1]+direction[1])
                        if self.can_move_map[target_pos[0], target_pos[1]] == 1:
                            pass
                        else:
                            action_idx = 1 + dir
                            valid_action_cart[action_idx] = 0.
                action = choose_action(valid_action_cart, logits)
                if action in [1,2,3,4]:
                    move_dir = action-1
                action = cart_action_mapping[action]
                if action["type"] == "transfer":
                    direction = action["act"][1]
                    pos_delta = directions_mapping[direction]
                    transfer_pos = (pos[0]+pos_delta[0], pos[1]+pos_delta[1])
                    if self.map_info['unit_'+str(team)][transfer_pos[0]][transfer_pos[1]]:
                        dest_id = self.map_info['unit_'+str(team)][transfer_pos[0]][transfer_pos[1]]
                        decision = action["act"][0](id=id,dest_id=dest_id,amount=2000)
                elif action["type"] == "move":
                    direction = directions_mapping[move_dir]
                    target_pos = (pos[0]+direction[0], pos[1]+direction[1])
                    self.can_move_map[target_pos[0], target_pos[1]] = 0
                    self.can_move_map[pos[0], pos[1]] = 1
                    decision = action["act"](id=id)
                else:
                    decision = action["act"](id=id)
                
            if decision:
                decisions.append(decision)
                
            # print(decisions)
            # print("-------------------")
        return decisions

    def get_valid_actions(self, idx):
        def valid_worker(loc):
            valid_action = np.zeros(self.worker_act_dim, dtype = np.float32)
            if self.map_info['act_worker_'+str(idx)][loc[0]][loc[1]]:
                uid = self.map_info['act_worker_'+str(idx)][loc[0]][loc[1]]
                valid_action[0] = 1. # none is always valid
                # Check if move is valid
                for i in range(4):
                    direction = directions_mapping[i]
                    target_pos = (loc[0]+direction[0], loc[1]+direction[1])
                    action_dim_shift = 1 # None
                    if 0 <= target_pos[0] < self.map_size and 0 <= target_pos[1] < self.map_size:
                        if self.map_info["city_"+str(idx)][target_pos]: # if own city
                            valid_action[i+action_dim_shift] = 1.
                        else: # if not own city: if empty can move, else cannot move
                            valid_action[i+action_dim_shift] = self.map_info["is_empty"][target_pos]
                # Check if transfer is valid
                for i in range(4):
                    direction = directions_mapping[i]
                    target_pos = (loc[0]+direction[0], loc[1]+direction[1])
                    if 0 <= target_pos[0] < self.map_size and 0 <= target_pos[1] < self.map_size:
                        if self.map_info["unit_"+str(idx)][target_pos[0]][target_pos[1]]: 
                            # TODO: check if the target unit has achieved the capacity
                            action_dim_shift = {"wood":5,"coal":9,"uranium":13}
                            for resource in ["wood","coal","uranium"]:
                                if self.unit_info[uid][resource] > 0:
                                    valid_action[i+action_dim_shift[resource]] = 1.
                # Check if pillage is valid
                if self.map_info["road"][loc] > 0 and not self.map_info["city_"+str(idx)][loc]:
                    valid_action[17] = 1.
                # Check if build is valid
                if self.unit_info[uid]["wood"] + self.unit_info[uid]["coal"] + self.unit_info[uid]["uranium"] >= 99.9 and \
                    (self.map_info["is_res"][loc] == 0 and self.map_info["city_0"][loc] == 0 and self.map_info["city_1"][loc] == 0):
                    valid_action[18] = 1.
            return valid_action
        def valid_city():
            valid_action = np.zeros(self.ct_act_dim, dtype = np.float32)
            valid_action[0] = 0. # none is always invalid
            if self.global_info["ct_count"][idx] > self.global_info["unit_count"][idx]:
                valid_action[1] = 1.
                valid_action[2] = 1.
            if self.global_info["rp"][idx] < 200:
                valid_action[3] = 1.
            # if sum(valid_action) == 0.:
            #     valid_action[0] = 1.
            return valid_action
        def valid_cart(loc):
            valid_action = np.zeros(self.cart_act_dim, dtype = np.float32)
            if self.map_info["act_cart_"+str(idx)][loc[0]][loc[1]]:
                uid = self.map_info["act_cart_"+str(idx)][loc[0]][loc[1]]
                valid_action[0] = 1. # None is always valid
                # Check if move is valid
                for i in range(4):
                    direction = directions_mapping[i]
                    target_pos = (loc[0]+direction[0], loc[1]+direction[1])
                    action_dim_shift = 1 # None
                    if 0 <= target_pos[0] < self.map_size and 0 <= target_pos[1] < self.map_size:
                        if self.map_info["city_"+str(idx)][target_pos] == 1:
                            valid_action[i+action_dim_shift] = 1.
                        else:
                            valid_action[i+action_dim_shift] = self.map_info["is_empty"][target_pos]
                # Check if transfer is valid
                for i in range(4):
                    direction = directions_mapping[i]
                    target_pos = (loc[0]+direction[0], loc[1]+direction[1])
                    if 0 <= target_pos[0] < self.map_size and 0 <= target_pos[1] < self.map_size:
                        if self.map_info["unit_"+str(idx)][target_pos[0]][target_pos[1]]:
                            # TODO: check if the target unit has achieved the capacity
                            action_dim_shift = {"wood":5,"coal":9,"uranium":13}
                            for resource in ["wood","coal","uranium"]:
                                if self.unit_info[uid][resource] > 0:
                                    valid_action[i+action_dim_shift[resource]] = 1.
                    # return valid_action
            return valid_action
        valid_workers = np.asarray([valid_worker((w,h)) for w in range(self.map_size) for h in range(self.map_size)])
        valid_carts = np.asarray([valid_cart((w,h)) for w in range(self.map_size) for h in range(self.map_size)])
        valid_city_action = valid_city()
        valid_cities = np.asarray([copy.deepcopy(valid_city_action) if self.map_info["act_ct_"+str(idx)][w][h] else np.zeros(self.ct_act_dim, dtype = np.float32) for w in range(self.map_size) for h in range(self.map_size)])
        return [valid_workers, valid_carts, valid_cities]
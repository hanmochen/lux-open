from typing import Tuple, List, Dict
import copy
import numpy as np

time_scale = False

class FeatureParser(object):
    """
    feature parser
    """

    def __init__(self, feature_param: Dict):
        """
        feature parser, initialize

        :param feature_param:  global params
        """

        """ global params """

        self.map_size = feature_param['map_size']  # square map size
        self.num_day = feature_param['num_day']  # total number of days
        self.num_hour = feature_param['num_hour']  # number of hours per day
        self.num_hour_daytime = feature_param['num_hour_daytime']  # number of daytime hours per day
        self.wood_regrow_ub = feature_param['wood_regrow_ub']  # wood regrows if less than a value
        self.cooldown_action_ub = feature_param['cooldown_action_ub']  # units can act only if cooldown less than a value
        self.resource_ub_worker = feature_param['resource_ub_worker']  # worker's resource capacity
        self.resource_ub_cart = feature_param['resource_ub_cart']  # cart's resource capacity
        self.cost_base_city = feature_param['cost_base_city']  # citytile's basic cost
        self.cost_save_city = feature_param['cost_save_city']  # citytile's saving cost due to adjacent ones
        self.research_point_coal = feature_param['research_point_coal']  # research point pre-requisite of coal
        self.research_point_urn = feature_param['research_point_urn']  # research point pre-requisite of uranium
        self.dict_fuel_value = feature_param['dict_fuel_value']  # fuel value, {'wood': , 'coal': , 'uranium': }
        self.cost_worker_city = feature_param['cost_worker_city']  # worker's cost in city each turn at night
        self.cost_worker_out = feature_param['cost_worker_out']  # worker's cost out of city each turn at night
        self.cost_cart_city = feature_param['cost_cart_city']  # cart's cost in city each turn at night
        self.cost_cart_out = feature_param['cost_cart_out']  # cart's cost out of city each turn at night

        """ feature """

        # meaning of each index
        # attention: suffix '_own' and '_enm' cannot rename, because used when feature change view
        self.dict_global_idx = {
            'day': {d: d for d in range(self.num_day)},  # bool, day is 0-based
            'hour': {h: self.num_day + h for h in range(self.num_hour)},  # bool, hour is 0-based
            'daytime_or_night': (self.num_day + self.num_hour, self.num_day + self.num_hour + 1),  # bool
            'num_citytile_own': self.num_day + self.num_hour + 2,  # int
            'num_citytile_enm': self.num_day + self.num_hour + 3,  # int
            'num_unit_own': self.num_day + self.num_hour + 4,  # int
            'num_unit_enm': self.num_day + self.num_hour + 5,  # int
            'research_point_own': self.num_day + self.num_hour + 6,  # int
            'research_point_enm': self.num_day + self.num_hour + 7,  # int
            'total_fuel_own': self.num_day + self.num_hour + 8,  # int
            'total_fuel_enm': self.num_day + self.num_hour + 9,  # int
            'total_cost_own': self.num_day + self.num_hour + 10,  # int
            'total_cost_enm': self.num_day + self.num_hour + 11,  # int
            'avr_fuel_own': self.num_day + self.num_hour + 12,  # float
            'avr_fuel_enm': self.num_day + self.num_hour + 13,  # float
            'avr_cost_own': self.num_day + self.num_hour + 14,  # float
            'avr_cost_enm': self.num_day + self.num_hour + 15,  # float
            'can_coal_own': self.num_day + self.num_hour + 16,  # bool
            'can_coal_enm': self.num_day + self.num_hour + 17,  # bool
            'can_urn_own': self.num_day + self.num_hour + 18,  # bool
            'can_urn_enm': self.num_day + self.num_hour + 19  # bool
        }
        self.dict_image_idx = {
            'worker_empty': 0,  # bool, no worker in the cell
            'worker_own': 1,  # bool, own worker in the cell
            'worker_enm': 2,  # bool, enemy worker in the cell
            'cart_empty': 3,  # bool
            'cart_own': 4,  # bool
            'cart_enm': 5,  # bool
            'city_empty': 6,  # bool
            'city_own': 7,  # bool
            'city_enm': 8,  # bool
            'road_level': 9,  # float
            'worker_cooldown': 10,  # float, choose minimum if multi
            'worker_can_act': 11,  # bool
            'cart_cooldown': 12,  # float, choose minimum if multi
            'cart_can_act': 13,  # bool
            'city_cooldown': 14,  # float
            'city_can_act': 15,  # bool
            'has_resource': 16,  # bool
            'num_wood_rsc': 17,  # int, resource wood amount
            'wood_regrow': 18,  # bool
            'num_coal_rsc': 19,  # int
            'num_urn_rsc': 20,  # int
            'num_wood_worker': 21,  # int, worker's wood amount
            'num_coal_worker': 22,  # int
            'num_urn_worker': 23,  # int
            'reach_ub_worker': 24,  # bool, if worker reaches resource upper bound
            'num_wood_cart': 25,  # int, cart's wood amount
            'num_coal_cart': 26,  # int
            'num_urn_cart': 27,  # int
            'reach_ub_cart': 28,  # bool
            'city_cost': 29,  # int
            'city_avr_fuel': 30,  # float, cost weighted average fuel
            'city_can_survive_tonight': 31,  # bool
            'city_fuel_needed': 32,  # float
            'worker_in_city': 33,  # bool
            'cart_in_city': 34,  # bool
            'x_delta': 35,  # int, distance from centre
            'y_delta': 36  # int
        }

        self.normalize_coefs = {
            'city_count': 100,
            'unit_count': 100,
            'total_fuel': 2300,
            'total_cost': 230,
            'avr_fuel': 230,
            'avr_cost': 23,
            'research_point': 200,
            'road_level':6,
            'cooldown': 10,
            'resource': 100,
            'distance': self.map_size//2,
        }

        # features
        self.global_feature = np.zeros(shape=self.num_day + self.num_hour + 2 + len(self.dict_global_idx) - 3)
        self.image_feature = np.zeros(shape=(len(self.dict_image_idx), self.map_size, self.map_size))

        # change view, own to enemy
        self.global_feature_op = np.zeros(shape=self.num_day + self.num_hour + 2 + len(self.dict_global_idx) - 3)
        self.image_feature_op = np.zeros(shape=(len(self.dict_image_idx), self.map_size, self.map_size))
        # time
        self.day, self.hour = 0, 0
        self.if_night = False

        # units
        # {unit id: {'type': 'worker' or 'cart', 'team': 0 or 1, 'loc': coordinate, 'cooldown': }}
        self.unit_info = {}

        # cities
        # {city id: {'team': , 'tiles': list of citytile coordinates, 'fuel': , 'cost': }}
        self.city_info = {}

        self.map_info = {
            'is_empty': np.ones(shape=(self.map_size, self.map_size)),  # bool, if empty
            'city_0': np.zeros(shape=(self.map_size, self.map_size)),  # bool, if own citytile
            'city_1': np.zeros(shape=(self.map_size, self.map_size)),  # bool, if enemy citytile
            'unit_0': [[0]*self.map_size for _ in range(self.map_size)],  # 0 or str, if own units
            'unit_1': [[0]*self.map_size for _ in range(self.map_size)],  # 0 or str, if enemy units
            'road': np.zeros(shape=(self.map_size, self.map_size)),  # float, road level
            'is_res': np.zeros(shape=(self.map_size, self.map_size)),  # if resource
            'act_ct_0': np.zeros(shape=(self.map_size, self.map_size)),  # if own citytile and can act
            'act_worker_0': [[0]*self.map_size for _ in range(self.map_size)],  # if own worker and can act
            'act_cart_0': [[0]*self.map_size for _ in range(self.map_size)],  # if own cart and can act
            'act_ct_1': np.zeros(shape=(self.map_size, self.map_size)),  # if enemy citytile and can act
            'act_worker_1': [[0]*self.map_size for _ in range(self.map_size)],  # if enemy worker and can act
            'act_cart_1': [[0]*self.map_size for _ in range(self.map_size)],  # if enemy cart and can act
        }


        self.global_info = {
            'unit_count': {0: 1, 1: 1},  # unit amount in each team
            'ct_count': {0: 1, 1: 1},  # city tile amount in each team
            'rp': {0: 0, 1: 0},  # research point of each team
            'is_night': False,  # if night current moment
            'total_fuel': {0: 0, 1: 0},  # total fuel of each team
        }

        self.task_mask = {0: [], 1: []}

    def reset(self):
        self.global_feature = np.zeros(shape=self.num_day + self.num_hour + 2 + len(self.dict_global_idx) - 3)
        self.image_feature = np.zeros(shape=(len(self.dict_image_idx), self.map_size, self.map_size))

        # change view, own to enemy
        self.global_feature_op = np.zeros(shape=self.num_day + self.num_hour + 2 + len(self.dict_global_idx) - 3)
        self.image_feature_op = np.zeros(shape=(len(self.dict_image_idx), self.map_size, self.map_size))
        # time
        self.day, self.hour = 0, 0
        self.if_night = False

        # units
        # {unit id: {'type': 'worker' or 'cart', 'team': 0 or 1, 'loc': coordinate, 'cooldown': }}
        self.unit_info = {}

        # cities
        # {city id: {'team': , 'tiles': list of citytile coordinates, 'fuel': , 'cost': }}
        self.city_info = {}

        self.map_info = {
            'is_empty': np.ones(shape=(self.map_size, self.map_size)),  # bool, if empty
            'city_0': np.zeros(shape=(self.map_size, self.map_size)),  # bool, if own citytile
            'city_1': np.zeros(shape=(self.map_size, self.map_size)),  # bool, if enemy citytile
            'unit_0': [[0]*self.map_size for _ in range(self.map_size)],  # 0 or str, if own units
            'unit_1': [[0]*self.map_size for _ in range(self.map_size)],  # 0 or str, if enemy units
            'road': np.zeros(shape=(self.map_size, self.map_size)),  # float, road level
            'is_res': np.zeros(shape=(self.map_size, self.map_size)),  # if resource
            'act_ct_0': np.zeros(shape=(self.map_size, self.map_size)),  # if own citytile and can act
            'act_worker_0': [[0]*self.map_size for _ in range(self.map_size)],  # if own worker and can act
            'act_cart_0': [[0]*self.map_size for _ in range(self.map_size)],  # if own cart and can act
            'act_ct_1': np.zeros(shape=(self.map_size, self.map_size)),  # if enemy citytile and can act
            'act_worker_1': [[0]*self.map_size for _ in range(self.map_size)],  # if enemy worker and can act
            'act_cart_1': [[0]*self.map_size for _ in range(self.map_size)],  # if enemy cart and can act
        }


        self.global_info = {
            'unit_count': {0: 1, 1: 1},  # unit amount in each team
            'ct_count': {0: 1, 1: 1},  # city tile amount in each team
            'rp': {0: 0, 1: 0},  # research point of each team
            'is_night': False,  # if night current moment
            'total_fuel': {0: 0, 1: 0},  # total fuel of each team
        }

        self.task_mask = {0: [], 1: []}

    def parse(self,obs: Dict) -> Tuple[List[np.ndarray], Dict, Dict, Dict]:
        """
        feature parser, run
        :param obs:  observation from environment
        :return: nothing
        """

        self._get_feature(obs=obs)
        self._normalize()
        self._feature_change_view()
        # self._get_task_mask()
        
        feature_own = self._flatten(global_feature=self.global_feature, image_feature=self.image_feature)
        feature_enm = self._flatten(global_feature=self.global_feature_op, image_feature=self.image_feature_op)
        return [feature_own, feature_enm], copy.deepcopy(
            self.unit_info), copy.deepcopy(self.map_info), copy.deepcopy(self.global_info)

    def _get_feature(self, obs: Dict):
        """
        get features, global and image
        :param obs:  observation from environment
        :return: nothing
        """

        # observation
        turn = obs['step']  # current turn
        if time_scale:
            if turn > 240:      
                turn = (turn - 240) //  2 + 240
        self.day, self.hour = turn // self.num_hour, turn % self.num_hour
        self.day = min(self.day,8)
        self.hours_to_go = self.num_hour - max(self.hour,self.num_hour_daytime)
        updates = obs['updates']

        # Reset 
        self.global_feature = np.zeros(shape=self.num_day + self.num_hour + 2 + len(self.dict_global_idx) - 3)
        self.image_feature = np.zeros(shape=(len(self.dict_image_idx), self.map_size, self.map_size))

        # units
        # {unit id: {'type': 'worker' or 'cart', 'team': 0 or 1, 'loc': coordinate, 'cooldown': }}
        self.unit_info = {}

        # cities
        # {city id: {'team': , 'tiles': list of citytile coordinates, 'fuel': , 'cost': }}
        self.city_info = {}

        self.map_info = {
            'is_empty': np.ones(shape=(self.map_size, self.map_size)),  # bool, if empty
            'city_0': np.zeros(shape=(self.map_size, self.map_size)),  # bool, if own citytile
            'city_1': np.zeros(shape=(self.map_size, self.map_size)),  # bool, if enemy citytile
            'unit_0': [[0]*self.map_size for _ in range(self.map_size)],  # 0 or str, if own units
            'unit_1': [[0]*self.map_size for _ in range(self.map_size)],  # 0 or str, if enemy units
            'road': np.zeros(shape=(self.map_size, self.map_size)),  # float, road level
            'is_res': np.zeros(shape=(self.map_size, self.map_size)),  # if resource
            'act_ct_0': np.zeros(shape=(self.map_size, self.map_size)),  # if own citytile and can act
            'act_worker_0': [[0]*self.map_size for _ in range(self.map_size)],  # if own worker and can act
            'act_cart_0': [[0]*self.map_size for _ in range(self.map_size)],  # if own cart and can act
            'act_ct_1': np.zeros(shape=(self.map_size, self.map_size)),  # if enemy citytile and can act
            'act_worker_1': [[0]*self.map_size for _ in range(self.map_size)],  # if enemy worker and can act
            'act_cart_1': [[0]*self.map_size for _ in range(self.map_size)],  # if enemy cart and can act
        }

        self.global_info = {
            'unit_count': {0: 1, 1: 1},  # unit amount in each team
            'ct_count': {0: 1, 1: 1},  # city tile amount in each team
            'rp': {0: 0, 1: 0},  # research point of each team
            'is_night': False,  # if night current moment
            'total_fuel': {0: 0, 1: 0},  # total fuel of each team
        } 

        # Feature Initialization
        self.image_feature[self.dict_image_idx['worker_empty'], :, :] = True
        self.image_feature[self.dict_image_idx['cart_empty'], :, :] = True
        self.image_feature[self.dict_image_idx['city_empty'], :, :] = True
        self.image_feature[self.dict_image_idx['worker_cooldown'], :, :] = 10
        self.image_feature[self.dict_image_idx['cart_cooldown'], :, :] = 10
        self.image_feature[self.dict_image_idx['city_cooldown'], :, :] = 10
        self.image_feature[self.dict_image_idx['city_can_survive_tonight'], :, :] = False
        # updates
        for info in updates:
            if info == "D_DONE":
                break

            info = info.split(' ')  # info in obs, string mode, split into list of strings

            # case: resource
            # 0: 'r'  1: resource type  2,3: x,y  4: amount
            if info[0] == 'r':
                info[2], info[3], info[4] = int(info[2]), int(info[3]), int(info[4])
                self.image_feature[self.dict_image_idx['has_resource'], info[2], info[3]] = True
                self.map_info['is_res'][info[2], info[3]] = True
                idx_rsc_type = self.dict_image_idx['num_wood_rsc'] if info[1] == 'wood' else self.dict_image_idx[
                    'num_coal_rsc'] if info[1] == 'coal' else self.dict_image_idx['num_urn_rsc']
                self.image_feature[idx_rsc_type, info[2], info[3]] = info[4]

                # if wood regrows
                self.image_feature[self.dict_image_idx['wood_regrow'], info[2], info[3]] = (info[1] == 'wood' and info[4] < self.wood_regrow_ub)

            # case: unit
            # 0: 'u'  1: unit type  2: team  3: unit id  4,5: x,y  6: cooldown  7,8,9: wood,coal,uranium
            if info[0] == 'u':
                info[1], info[2], info[4], info[5], info[6], info[7], info[8], info[9] = int(info[1]), int(
                    info[2]), int(info[4]), int(info[5]), float(info[6]), int(info[7]), int(info[8]), int(info[9])

                # add to unit info
                self.unit_info[info[3]] = {'type': 'worker' if info[1]==0 else 'cart',  # 0: worker  1: cart
                                           'team': info[2], 
                                           'loc': (info[4], info[5]),
                                           'cooldown': info[6],
                                           'wood': info[7],
                                           'coal': info[8],
                                           'uranium': info[9]}

                self.map_info['unit_'+str(info[2])][info[4]][info[5]] = info[3]
                self.map_info['is_empty'][info[4],info[5]] = 0
                # worker
                if info[1]==0:

                    # Team 0 view
                    self.image_feature[self.dict_image_idx['worker_empty'], info[4], info[5]] = False

                    # team
                    if info[2]==0: # team 0
                        self.image_feature[self.dict_image_idx['worker_own'], info[4], info[5]] = True
                        self.global_feature[self.dict_global_idx["num_unit_own"]] += 1
                    else:
                        self.image_feature[self.dict_image_idx['worker_enm'], info[4], info[5]] = True
                        self.global_feature[self.dict_global_idx["num_unit_enm"]] += 1

                    # cooldown
                    self.image_feature[self.dict_image_idx['worker_cooldown'], info[4], info[5]] = min(info[6], 
                        self.image_feature[self.dict_image_idx['worker_cooldown'], info[4], info[5]]) 

                    if self.image_feature[self.dict_image_idx['worker_cooldown'], info[4], info[5]] < self.cooldown_action_ub:
                        self.image_feature[self.dict_image_idx['worker_can_act'], info[4], info[5]] = 1
                        self.map_info['act_worker_'+str(info[2])][info[4]][info[5]] = info[3]

                    # resource
                    self.image_feature[self.dict_image_idx['num_wood_worker'], info[4], info[5]] = info[7]
                    self.image_feature[self.dict_image_idx['num_coal_worker'], info[4], info[5]] = info[8]
                    self.image_feature[self.dict_image_idx['num_urn_worker'], info[4], info[5]] = info[9]
                    self.image_feature[self.dict_image_idx['reach_ub_worker'], info[4], info[5]] = (
                            info[7] + info[8] + info[9] >= self.resource_ub_worker)
                    self.global_info['total_fuel'][info[2]] += info[7]*1 + info[8]*10 + info[9]*40

                    
                # cart
                else:
                    self.image_feature[self.dict_image_idx['cart_empty'], info[4], info[5]] = False
                    # team
                    if info[2]==0:
                        self.image_feature[self.dict_image_idx['cart_own'], info[4], info[5]] = True
                        self.global_feature[self.dict_global_idx['num_unit_own']] += 1
                    else:
                        self.image_feature[self.dict_image_idx['cart_enm'], info[4], info[5]] = True
                        self.global_feature[self.dict_global_idx['num_unit_enm']] += 1

                    # cooldown
                    self.image_feature[self.dict_image_idx['cart_cooldown'], info[4], info[5]] = min(info[6],
                        self.image_feature[self.dict_image_idx['cart_cooldown'], info[4], info[5]])

                    if self.image_feature[self.dict_image_idx['cart_cooldown'],info[4], info[5]] < self.cooldown_action_ub:
                        self.image_feature[self.dict_image_idx['cart_can_act'], info[4], info[5]] = 1
                        self.map_info['act_cart_'+str(info[2])][info[4]][info[5]] = info[3]

                    # resource
                    self.image_feature[self.dict_image_idx['num_wood_cart'], info[4], info[5]] = info[7]
                    self.image_feature[self.dict_image_idx['num_coal_cart'], info[4], info[5]] = info[8]
                    self.image_feature[self.dict_image_idx['num_urn_cart'], info[4], info[5]] = info[9]
                    self.image_feature[self.dict_image_idx['reach_ub_cart'], info[4], info[5]] = (
                            info[7] + info[8] + info[9] >= self.resource_ub_cart)
                    self.global_info['total_fuel'][info[2]] += info[7]*1 + info[8]*10 + info[9]*40

            # case: city
            # 0: 'c'  1: team  2: city id  3: total fuel  4: total cost each turn
            if info[0] == 'c':
                info[1], info[3], info[4] = int(info[1]), int(info[3]), int(info[4])
                self.city_info[info[2]] = {'team': info[1], 'tiles': [], 'fuel': info[3], 'cost': info[4]}
                if info[1] == 0:
                    self.global_feature[self.dict_global_idx['total_fuel_own']] += info[3]
                    self.global_feature[self.dict_global_idx['total_cost_own']] += info[4]
                else:
                    self.global_feature[self.dict_global_idx['total_fuel_enm']] += info[3]
                    self.global_feature[self.dict_global_idx['total_cost_enm']] += info[4]

            # case: citytile
            # 0: 'ct'  1: team  2: city id  3,4: x,y  5: cooldown
            if info[0] == 'ct':
                info[1], info[3], info[4], info[5] = int(info[1]), int(info[3]), int(info[4]), float(info[5])
                self.image_feature[self.dict_image_idx['city_empty'], info[3], info[4]] = False
                self.map_info['city_'+str(info[1])][info[3],info[4]] = 1
                self.map_info['is_empty'][info[3],info[4]] = 0

                # team
                if info[1]==0:
                    self.image_feature[self.dict_image_idx['city_own'], info[3], info[4]] = True
                    self.global_feature[self.dict_global_idx['num_citytile_own']] += 1
                else:
                    self.image_feature[self.dict_image_idx['city_enm'], info[3], info[4]] = True
                    self.global_feature[self.dict_global_idx['num_citytile_enm']] += 1

                # cooldown
                self.image_feature[self.dict_image_idx['city_cooldown'], info[3], info[4]] = info[5]
                if (info[5] < self.cooldown_action_ub):
                    self.image_feature[self.dict_image_idx['city_can_act'], info[3], info[4]] = 1
                    self.map_info['act_ct_'+str(info[1])][info[3],info[4]] = 1

                # city affiliation: copy coordinate to city dictionary
                self.city_info[info[2]]['tiles'].append((info[3], info[4]))


            # case: road level
            # 0: 'ccd'  1,2: x,y  3: road level
            if info[0] == 'ccd':
                info[1], info[2], info[3] = int(info[1]), int(info[2]), float(info[3])
                self.image_feature[self.dict_image_idx['road_level'], info[1], info[2]] = info[3]
                self.map_info['road'][info[1],info[2]] = info[3]

            # case: research point
            # 0: 'rp'  1: team  2: value
            if info[0] == 'rp':
                info[1], info[2] = int(info[1]), int(info[2])
                if info[1]==0:
                    self.global_feature[self.dict_global_idx['research_point_own']] = info[2]
                else:
                    self.global_feature[self.dict_global_idx['research_point_enm']] = info[2]

        # cities
        for city in self.city_info:
            for tile in self.city_info[city]['tiles']:
                # cost and weighted average fuel
                num_adjacent = 0
                list_adjacent = self._get_adjacent(x=tile[0], y=tile[1])
                if self.city_info[city]['team'] == 0:
                    num_adjacent = np.sum(self.image_feature[self.dict_image_idx['city_own']][list_adjacent])
                else:
                    num_adjacent = np.sum(self.image_feature[self.dict_image_idx['city_enm']][list_adjacent])

                cost = self.cost_base_city - self.cost_save_city * num_adjacent
                avr_fuel = self.city_info[city]['fuel'] * (cost / self.city_info[city]['cost'])
                self.image_feature[self.dict_image_idx['city_cost'], tile[0], tile[1]] = cost
                self.image_feature[self.dict_image_idx['city_avr_fuel'], tile[0], tile[1]] = avr_fuel
                self.image_feature[self.dict_image_idx['city_can_survive_tonight'], tile[0], tile[1]] = (avr_fuel >= self.hours_to_go * cost)
                self.image_feature[self.dict_image_idx['city_fuel_needed'], tile[0], tile[1]] = max(0,self.hours_to_go*cost-avr_fuel)

                # workers and carts
                self.image_feature[self.dict_image_idx['worker_in_city'], tile[0], tile[1]] = not self.image_feature[self.dict_image_idx['worker_empty'], tile[0], tile[1]]
                self.image_feature[self.dict_image_idx['cart_in_city'], tile[0], tile[1]] = not self.image_feature[self.dict_image_idx['cart_empty'], tile[0], tile[1]]

        # distance to centre
        centre = (self.map_size//2 - 0.5, self.map_size//2 - 0.5) # 15.5,15.5
        for x in range(self.map_size):
            for y in range(self.map_size):
                self.image_feature[self.dict_image_idx['x_delta'], x, y] = abs(x - centre[0])
                self.image_feature[self.dict_image_idx['y_delta'], x, y] = abs(y - centre[1])

        # time
        
        self.global_feature[self.dict_global_idx['day'][self.day]] = 1
        self.global_feature[self.dict_global_idx['hour'][self.hour]] = 1
        self.if_night = (self.hour >= self.num_hour_daytime)
        self.global_feature[self.dict_global_idx['daytime_or_night'][0]] = not self.if_night # is day
        self.global_feature[self.dict_global_idx['daytime_or_night'][1]] = self.if_night # is night

        # other global info

        self.global_feature[self.dict_global_idx['avr_fuel_own']] = self.global_feature[self.dict_global_idx['total_fuel_own']] / self.global_feature[self.dict_global_idx['num_citytile_own']] if self.global_feature[self.dict_global_idx['num_citytile_own']] else 0
        self.global_feature[self.dict_global_idx['avr_fuel_enm']] = self.global_feature[self.dict_global_idx['total_fuel_enm']] / self.global_feature[self.dict_global_idx['num_citytile_enm']] if self.global_feature[self.dict_global_idx['num_citytile_enm']] else 0
        self.global_feature[self.dict_global_idx['avr_cost_own']] = self.global_feature[self.dict_global_idx['total_cost_own']] / self.global_feature[self.dict_global_idx['num_citytile_own']] if self.global_feature[self.dict_global_idx['num_citytile_own']] else 0
        self.global_feature[self.dict_global_idx['avr_cost_enm']] = self.global_feature[self.dict_global_idx['total_cost_enm']] / self.global_feature[self.dict_global_idx['num_citytile_enm']] if self.global_feature[self.dict_global_idx['num_citytile_enm']] else 0


        self.global_feature[self.dict_global_idx['can_coal_own']] = (self.global_feature[self.dict_global_idx['research_point_own']] >= self.research_point_coal) 
        self.global_feature[self.dict_global_idx['can_coal_enm']] = (self.global_feature[self.dict_global_idx['research_point_enm']] >= self.research_point_coal)
        self.global_feature[self.dict_global_idx['can_urn_own']] = (self.global_feature[self.dict_global_idx['research_point_own']] >= self.research_point_urn) 
        self.global_feature[self.dict_global_idx['can_coal_enm']] = (self.global_feature[self.dict_global_idx['research_point_enm']] >= self.research_point_coal)


        # units amount, city tile amount and research point
        self.global_info['unit_count'][0] = self.global_feature[self.dict_global_idx['num_unit_own']]
        self.global_info['unit_count'][1] = self.global_feature[self.dict_global_idx['num_unit_enm']]
        self.global_info['ct_count'][0] = self.global_feature[self.dict_global_idx['num_citytile_own']]
        self.global_info['ct_count'][1] = self.global_feature[self.dict_global_idx['num_citytile_enm']]
        self.global_info['rp'][0] = self.global_feature[self.dict_global_idx['research_point_own']]
        self.global_info['rp'][1] = self.global_feature[self.dict_global_idx['research_point_enm']]
        self.global_info['total_fuel'][0] += self.global_feature[self.dict_global_idx['total_fuel_own']]
        self.global_info['total_fuel'][1] += self.global_feature[self.dict_global_idx['total_fuel_enm']]
        self.global_info['is_night'] = self.if_night
        self.global_info['map_size'] = self.map_size

    def _get_adjacent(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        get adjacent coordinates in the map
        :param x:  x coordinate
        :param y:  y coordinate
        :return:  list of adjacent coordinates
        """

        xs, ys = [], []
        if x:
            xs.append(x-1)
            ys.append(y)
        if x+1<self.map_size:
            xs.append(x+1)
            ys.append(y)
        if y:
            xs.append(x)
            ys.append(y-1)
        if y+1<self.map_size:
            xs.append(x)
            ys.append(y+1)
        return tuple(xs),tuple(ys)

    def _feature_change_view(self):
        """
        get feature in enemy view
        :return: nothing
        """

        # global feature
        self.global_feature_op = copy.deepcopy(self.global_feature)
        for key in self.dict_global_idx:
            if key[-4:] == '_own':
                op_key = key[:-4] + '_enm'
                self.global_feature_op[self.dict_global_idx[key]], self.global_feature_op[
                    self.dict_global_idx[op_key]]= self.global_feature_op[
                        self.dict_global_idx[op_key]], self.global_feature_op[self.dict_global_idx[key]]

        # image feature
        self.image_feature_op = copy.deepcopy(self.image_feature)

        for key in self.dict_image_idx:
            if key[-4:] == '_own':
                op_key = key[:-4] + '_enm'
                self.image_feature_op[self.dict_image_idx[key]], self.image_feature_op[
                    self.dict_image_idx[op_key]] = copy.deepcopy(self.image_feature_op[
                        self.dict_image_idx[op_key]]), copy.deepcopy(self.image_feature_op[self.dict_image_idx[key]])

    def _normalize(self):

        # normalize global feature
        self.global_feature[self.dict_global_idx['num_citytile_own']] = self.global_feature[self.dict_global_idx['num_citytile_own']] / self.normalize_coefs['city_count']
        self.global_feature[self.dict_global_idx['num_citytile_enm']] = self.global_feature[self.dict_global_idx['num_citytile_enm']] / self.normalize_coefs['city_count']
        self.global_feature[self.dict_global_idx['num_unit_own']] = self.global_feature[self.dict_global_idx['num_unit_own']] / self.normalize_coefs['unit_count']
        self.global_feature[self.dict_global_idx['num_unit_enm']] = self.global_feature[self.dict_global_idx['num_unit_enm']] / self.normalize_coefs['unit_count']
        self.global_feature[self.dict_global_idx['research_point_own']] = self.global_feature[self.dict_global_idx['research_point_own']] / self.normalize_coefs['research_point']
        self.global_feature[self.dict_global_idx['research_point_enm']] = self.global_feature[self.dict_global_idx['research_point_enm']] / self.normalize_coefs['research_point']
        self.global_feature[self.dict_global_idx['total_cost_own']] = self.global_feature[self.dict_global_idx['total_cost_own']] / self.normalize_coefs['total_cost']
        self.global_feature[self.dict_global_idx['total_cost_enm']] = self.global_feature[self.dict_global_idx['total_cost_enm']] / self.normalize_coefs['total_cost']
        self.global_feature[self.dict_global_idx['total_fuel_own']] = self.global_feature[self.dict_global_idx['total_fuel_own']] / self.normalize_coefs['total_fuel']
        self.global_feature[self.dict_global_idx['total_fuel_enm']] = self.global_feature[self.dict_global_idx['total_fuel_enm']] / self.normalize_coefs['total_fuel']
        self.global_feature[self.dict_global_idx['avr_cost_own']] = self.global_feature[self.dict_global_idx['avr_cost_own']] / self.normalize_coefs['avr_cost']
        self.global_feature[self.dict_global_idx['avr_cost_enm']] = self.global_feature[self.dict_global_idx['avr_cost_enm']] / self.normalize_coefs['avr_cost']
        self.global_feature[self.dict_global_idx['avr_fuel_own']] = self.global_feature[self.dict_global_idx['avr_fuel_own']] / self.normalize_coefs['avr_fuel']
        self.global_feature[self.dict_global_idx['avr_fuel_enm']] = self.global_feature[self.dict_global_idx['avr_fuel_enm']] / self.normalize_coefs['avr_fuel']


        # normalize image feature
        # Fix(jiaxin) changes the cooldown feature maps
        self.image_feature[self.dict_image_idx['road_level']] = self.image_feature[self.dict_image_idx['road_level']] / self.normalize_coefs['road_level']
        self.image_feature[self.dict_image_idx['worker_cooldown']] = 1 - self.image_feature[self.dict_image_idx['worker_cooldown']] / self.normalize_coefs['cooldown']
        self.image_feature[self.dict_image_idx['cart_cooldown']] = 1 - self.image_feature[self.dict_image_idx['cart_cooldown']] / self.normalize_coefs['cooldown']
        self.image_feature[self.dict_image_idx['city_cooldown']] = 1 - self.image_feature[self.dict_image_idx['city_cooldown']] / self.normalize_coefs['cooldown'] 
        self.image_feature[self.dict_image_idx['num_wood_rsc']] = self.image_feature[self.dict_image_idx['num_wood_rsc']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_coal_rsc']] = self.image_feature[self.dict_image_idx['num_coal_rsc']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_urn_rsc']] = self.image_feature[self.dict_image_idx['num_urn_rsc']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_wood_worker']] = self.image_feature[self.dict_image_idx['num_wood_worker']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_coal_worker']] = self.image_feature[self.dict_image_idx['num_coal_worker']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_urn_worker']] = self.image_feature[self.dict_image_idx['num_urn_worker']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_wood_cart']] = self.image_feature[self.dict_image_idx['num_wood_cart']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_coal_cart']] = self.image_feature[self.dict_image_idx['num_coal_cart']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['num_urn_cart']] = self.image_feature[self.dict_image_idx['num_urn_cart']] / self.normalize_coefs['resource']
        self.image_feature[self.dict_image_idx['city_cost']] = self.image_feature[self.dict_image_idx['city_cost']] / self.normalize_coefs['avr_cost']
        self.image_feature[self.dict_image_idx['city_avr_fuel']] = self.image_feature[self.dict_image_idx['city_avr_fuel']] / self.normalize_coefs['avr_fuel']
        self.image_feature[self.dict_image_idx['city_fuel_needed']] = self.image_feature[self.dict_image_idx['city_fuel_needed']] / self.normalize_coefs['avr_fuel']
        self.image_feature[self.dict_image_idx['x_delta']] = self.image_feature[self.dict_image_idx['x_delta']] / self.normalize_coefs['distance']
        self.image_feature[self.dict_image_idx['y_delta']] = self.image_feature[self.dict_image_idx['y_delta']] / self.normalize_coefs['distance']

    def _flatten(self, global_feature: np.ndarray, image_feature: np.ndarray) -> np.ndarray:
        """
        rearrange global and image feature to a single dimension np.ndarray
        :param global_feature:
        :param image_feature:
        :return:  rearranged np.ndarray
        """
        return np.concatenate((global_feature.flatten(), image_feature.flatten()))

    def _get_task_mask(self):
        """
        get the task mask for the current state
        :return:
        """
        self.task_mask = {0: [], 1: []}
        
        for team in [0,1]:
            map_info_key = lambda unit_type: '_'.join(['act',unit_type,str(team)])
            task_mask = []
            for unit_type in ['worker','cart','ct']:
                unit_mask = np.zeros(self.map_size*self.map_size)
                for w in range(self.map_size):
                    for h in range(self.map_size):
                        if self.map_info[map_info_key(unit_type)][w][h]:
                            unit_mask[w*self.map_size+h] = 1
                task_mask.append(unit_mask)
            self.task_mask[team] = task_mask


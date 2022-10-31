import copy
import random
from typing import DefaultDict
from pprint import pprint

import os, sys
ANC_LEVEL = 3
PARRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for i in range(ANC_LEVEL):
    sys.path.append(PARRENT_DIR)
    PARRENT_DIR = os.path.dirname(PARRENT_DIR)

import numpy as np
from numpy.lib.function_base import corrcoef
np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=30, linewidth=200)
np.printoptions(precision=3, suppress=True)


DTYPE = np.float32
U_DIM = 9
CT_DIM = 9

def dupli(item, times):
    _ls = []
    for _ in range(times):
        ditem = copy.deepcopy(item)
        _ls.append(ditem)
    return _ls

def oh(n, l):
    assert n < l
    assert isinstance(n, int)
    res = np.zeros(l, dtype=DTYPE)
    res[n] = 1
    return res

def unnestedl2(ls):
    res = []
    for it in ls:
        if isinstance(it, list):
            res.extend(it)
        else:
            res.append(it)
    return res

def _center_to(array, xc, yc):
    big_array = np.zeros((array.shape[0]+2*WH_RF, array.shape[1]+2*HH_RF))
    big_array[WH_RF:WH_RF+array.shape[0], HH_RF:HH_RF+array.shape[1]] = array
    return big_array[xc:xc+W_RF, yc:yc+H_RF]

def center_to(dmap, xc, yc):
    new_dmap = None
    if isinstance(dmap, dict):
        new_dmap = {}
        for key in dmap.keys():
            new_dmap[key] = [_center_to(_map, xc, yc) for _map in dmap[key]]
    if isinstance(dmap, list):
        new_dmap = []
        for _map in dmap:
            new_dmap.append(_center_to(_map, xc, yc))
    return new_dmap
    
class DeFeatureParser(object):
    def __init__(self, localmap_size, is_trans, max_citytile_num=160, max_worker_num=120, width=32, height=32):
        global WH_RF
        global HH_RF
        global W_RF
        global H_RF
        global O_DIM
        
        WH_RF = int(localmap_size)//2 # width half receptive field
        HH_RF = int(localmap_size)//2 # height half receptive field
        W_RF = 2 * WH_RF + 1 # width receptive field
        H_RF = 2 * HH_RF + 1 # height receptive field
        O_DIM = 6863 + 17 * localmap_size ** 2
        assert W_RF == localmap_size, 'localmap_size must be odd number.'

        super(DeFeatureParser, self).__init__()
        self.is_trans = is_trans
        self.max_citytile_num = max_citytile_num
        self.max_worker_num = max_worker_num
        self.max_cart_num = max_citytile_num - max_worker_num
        self.feature_dim = None
        self.width = width
        self.height = height
        self.origin_width = width
        self.origin_height = height
        
        #self.city_attr = DefaultDict(lambda: [0, 0])
        self.prev_ids = []
        self.GLOBAL_PID = 0
        self.UID2PID = {}
        self.PID2UID = {}
        self.orgin_quads = {}
        self.dims = None
        self.featuregb_dims = {}
        self.self_vec_dims = {}
        self.res1d_dims = {}
        self.r_dmap_dims = {}
        self.u_dmap_dims = {}
        self.fac_dmap_dims = {}
        self.edg_map_dims = {}
        self.map_infos = {}
        self.reset_infos()

        self.deadpool = set()
        self.game_state = None
        self.get_tct_count = lambda team, city_id: len(self.game_state.players[team].cities[city_id].citytiles)

    def reset_infos(self):
        #self.id_info = DefaultDict(lambda: DefaultDict(lambda: 0))
        self.id_info = DefaultDict(lambda: {
            'is_alive': 0,
            'loc': {0: [None, None], 1:[None, None]},
            'type': None,
            'team': None,
            'uid': None,
            'wood_carry': 0,
            'coal_carry': 0,
            'uranium_carry': 0,
            'cooldown': 0,
            'fuel': 0,
            'lightupkeep': 0,
            'at_city': 0, 
            'nights': 0, 
            'hunger': None,
        })

        self.city_info = DefaultDict(lambda: {
            'fuel': 0,
            'lightupkeep': 0,
            'tct_count': 0
        })

        self.global_info = {
            'step': None,
            'ct_count': {0: 0, 1: 0},
            'unit_count': {0: 0, 1: 0},
            'rp': {0: 0, 1: 0},
            'is_night': 0,
            'map_size': [32, 32],
            'orgin_quads': self.orgin_quads,
            'city_info': None
        }
    
    def get_quad(self, pos):
        '''
        x is column index, y is row index
        1 2 3 4 quadrants, 
        0<x<16, 0<y<16: 2 quadrant
        0<x<16, 16<=y<32: 3 quadrant
        16<=x<32, 0<y<16: 4 quadrant
        16<=x<32, 16<=y<32: 1 quadrant
        '''
        x, y = self.coord(pos[0], pos[1])
        if 0 <= x < 16 and 0 <= y < 16:
            return 2
        elif 0 <= x < 16 and 16 <= y < 32:
            return 3
        elif 16 <= x < 32 and 16 <= y < 32:
            return 4
        else:
            return 1

    def coord_trans(self, x, y, team, quads): 
        # team is 0/1, quads is a dict {0: 1:} 
        # quads are the orgin quads instead of the transformed quads 
        # (transformed quads are all self 1 op 4)

        quad = quads[team]
        op_team = 1 - team
        w = self.width - 1
        h = self.height - 1
        assert isinstance(op_team, int)

        op_quad = quads[op_team]

        if quad == 2 and op_quad == 1: 
            return h - y, x
        elif quad == 2 and op_quad == 3:
            return w - x, y
        elif quad == 3 and op_quad == 2:
            return w - x, h - y 
        elif quad == 3 and op_quad == 4:
            return y, x
        elif quad == 4 and op_quad == 1:
            return x, h - y
        elif quad == 4 and op_quad == 3:
            return y, w - x
        elif quad == 1 and op_quad == 4:
            return x, y        
        elif quad == 1 and op_quad == 2:
            return h - y, w - x
        else:
            raise ValueError

        # ------------------------------
        # if quad == 1 and op_quad == 4:
        # 0
        # xc, yc
        # (x-w/2)+w/2, (y-h/2)+h/2
        # x, y
        # e = [1, 0] : [1, 0] = e
        # w = [-1,0] : [-1,0] = w
        # n = [0,-1] : [0,-1] = n
        # s = [0, 1] : [0, 1] = s
        
        # if quad == 4 and op_quad == 1:
        # 0 + x-axis
        # xc, -yc
        # (x-w/2)+w/2, -(y-h/2)+h/2
        # x, h-y
        # e = [1, 0] : [1, 0] = e
        # w = [-1,0] : [-1,0] = w
        # n = [0,-1] : [0, 1] = s
        # s = [0, 1] : [0,-1] = n

        # ------------------------------
        # elif quad == 4 and op_quad == 3:
        # 90
        # yc, -xc
        #(y-h/2)+h/2, -(x-w/2)+w/2
        # y, w-x
        # e = [1, 0] : [0, -1] = n
        # w = [-1,0] : [0, 1] = s
        # n = [0,-1] : [-1, 0] = w
        # s = [0, 1] : [1, 0] = e
        
        # elif quad == 3 and op_quad == 4:
        # 90 + x-axis
        # yc, xc
        #(y-h/2)+h/2, (x-w/2)+w/2
        # y, x
        # e = [1, 0] : [0, 1] = s
        # w = [-1,0] : [0,-1] = n
        # n = [0,-1] : [-1,0] = w
        # s = [0, 1] : [1, 0] = e

        # ------------------------------
        # elif quad == 3 and op_quad == 2:
        # 180
        # -xc, -yc
        # -(x-w/2)+w/2, -(y-h/2)+h/2
        # w-x, h-y
        # e = [1, 0] : [-1, 0] = w
        # w = [-1,0] : [1,0] = e
        # n = [0,-1] : [0, 1] = s
        # s = [0, 1] : [0,-1] = n

        # elif quad == 2 and op_quad == 3:
        # 180 + x-axis
        # -xc, yc
        # -(x-w/2)+w/2, (y-h/2)+h/2
        # w-x, y
        # e = [1, 0] : [-1,0] = w
        # w = [-1,0] : [1, 0] = e
        # n = [0,-1] : [0,-1] = n
        # s = [0, 1] : [0, 1] = s

        # ------------------------------
        # elif quad == 2 and op_quad == 1:
        # -90
        # -yc, xc
        # -(y-h/2)+h/2, (x-w/2)+w/2
        # h-y, x
        # e = [1, 0] : [0, 1] = s
        # w = [-1,0] : [0,-1] = n
        # n = [0,-1] : [1, 0] = e
        # s = [0, 1] : [-1,0] = w

        # elif quad == 1 and op_quad == 2:
        # -90 + axis
        # -yc, -xc
        # -(y-h/2)+h/2, -(x-w/2)+w/2
        # h-y, w-x
        # e = [1, 0] : [0,-1] = n
        # w = [-1,0] : [0, 1] = s
        # n = [0,-1] : [1, 0] = e
        # s = [0, 1] : [-1,0] = w

        

    def coord(self, x, y):
        x += int((self.width - self.origin_width)/2)
        y += int((self.height - self.origin_height)/2)
        return x, y

    def gen_pid(self, uid):
        assert isinstance(uid, str)
        try:
            pid = self.UID2PID[uid]
        except:
            pid = self.GLOBAL_PID
            self.GLOBAL_PID += 1
            self.PID2UID[pid] = uid
            self.UID2PID[uid] = pid
        return pid
        
    def parse(self, obs, game_state, team_id):
        self.game_state = game_state
        self.reset_infos()
        _obs = obs
        team_ids = [team_id]
        self.origin_width, self.origin_height = game_state.map_width, game_state.map_height
        self.global_info['map_size'] = [self.origin_width, self.origin_height]

        res = {}
        team_features = {}

        for team in team_ids:
            featuregb = self.to_featuregb(team, _obs)
            res1d, r_dmap, u_dmap, fac_dmap, _map_info = self.to_feature(team, _obs['updates'])
            self.map_infos[team] = copy.deepcopy(_map_info)
            team_features[team] = [featuregb, res1d, r_dmap, u_dmap, fac_dmap]

            # update obs_list_len
            featuregb[0][0] = self.GLOBAL_PID

        

        for id in self.id_info.keys():
            if self.id_info[id]['team'] == team_id and self.id_info[id]['is_alive'] and self.id_info[id]['cooldown'] < 0.999 and self.id_info[id]['type'] == 'worker':
                featuregb, res1d, r_dmap, u_dmap, fac_dmap = team_features[team_id]
                self_vec, edg_map = self.to_featureself(id, res1d)
                x, y = self.id_info[id]['loc'][self.id_info[id]['team']]

                # compute dist
                for k in res1d:
                    xk, yk = res1d[k]['loc']
                    res1d[k]['dist'][0] = abs(x - xk) + abs(y - yk)

                res[id] = {
                    'featuregb': copy.deepcopy(featuregb),
                    'self_vec': copy.deepcopy(self_vec),
                    'res1d': copy.deepcopy(res1d),
                    'r_dmap': copy.deepcopy(center_to(r_dmap, x, y)),
                    'u_dmap': copy.deepcopy(center_to(u_dmap, x, y)),
                    'fac_dmap': copy.deepcopy(center_to(fac_dmap, x, y)),
                    'edg_map': copy.deepcopy(center_to(edg_map, x, y)),
                    }


        self.deadpool.update(set(self.prev_ids).difference(set(self.id_info.keys())))
        self.prev_ids = list(self.id_info.keys())
        

        # convert res to list of numpy
        res_n = self.res_normalize(res) # inplace!!!
        obs_list_n = self.res2obs_list(res_n)

        # add city_info as a nested dict to global_info
        self.global_info['city_info'] = self.city_info

        return obs_list_n, self.id_info, self.map_infos[team_id], self.global_info


    def res_normalize(self, res):
        #resc = copy.deepcopy(res) # Time consuming!!!!
        resc = res
        max_carry = {'worker': 1/100, 'cart': 1/1000, 'citytile': 0}
        for key in resc.keys():
            self_type = self.id_info[key]['type']
            # ignore the dead ids
            if key in self.deadpool: continue
            _res = resc[key]
            _res['featuregb'][0] *= 1/100 # obs_list_len
            _res['featuregb'][-3] *= 1/100 # rp

            if np.linalg.norm(_res['self_vec']['loc']): _res['self_vec']['loc'] *= 1/np.linalg.norm(_res['self_vec']['loc']) 
            _res['self_vec']['wood_carry'] *= max_carry[self_type]
            _res['self_vec']['coal_carry'] *= max_carry[self_type]
            _res['self_vec']['uranium_carry'] *= max_carry[self_type]
            _res['self_vec']['fuel'] *= 1/1000 # unbounded

            for k in _res['res1d'].keys():
                _type = self.id_info[k]['type']
                _res['res1d'][k]['dist'][0] *= 1/62

                if np.linalg.norm(_res['res1d'][k]['loc']): _res['res1d'][k]['loc'] *= 1/np.linalg.norm(_res['res1d'][k]['loc'])
                try:
                    _res['res1d'][k]['wood_carry'] *= max_carry[_type]
                    _res['res1d'][k]['coal_carry'] *= max_carry[_type]
                    _res['res1d'][k]['uranium_carry'] *= max_carry[_type]
                except:
                    _res['res1d'][k]['cooldown'] *= max_carry[_type]
                    _res['res1d'][k]['fuel'] *= 1/1000
                    _res['res1d'][k]['lightupkeep'] *= 1/23
                    # _res['res1d'][k]['nights'] *= 1/9
                    _res['res1d'][k]['hunger'] *= 1/1000
            
            _res['r_dmap']['wood'][0] *= 1/1000
            _res['r_dmap']['coal'][0] *= 1/1000
            _res['r_dmap']['uranium'][0] *= 1/1000
            
            for k in _res['u_dmap'].keys():
                _res['u_dmap'][k][2] *= [1/100, 1/1000][k]  # total_carry

            _res['fac_dmap']['ct'][2] *= 1/1000  # fuel # unbounded
            _res['fac_dmap']['ct'][3] *= 1/23 # lightupkeep
            # _res['fac_dmap']['ct'][4] *= 1/9  # nights # unbounded
            _res['fac_dmap']['ct'][5] *= 1/1000 # hunger # unbounded
            _res['fac_dmap']['ccd'][0] *= 1/6 # road

        return resc


    def res2obs_list(self, res):
        # Relying on python feature that dict keeps insertion order
        obs_list = [np.zeros(O_DIM, dtype=DTYPE)] * self.GLOBAL_PID
        for id in res.keys():
            featuregb_flatten = np.concatenate(res[id]['featuregb'])
            self_vec_flatten = np.concatenate(list(res[id]['self_vec'].values()))
            res1d_flatten = np.concatenate(self.pack_res1d(res[id]['res1d']))
            r_dmap_flatten = np.concatenate(unnestedl2(list(res[id]['r_dmap'].values()))).reshape(-1)
            u_dmap_flatten = np.concatenate(unnestedl2(list(res[id]['u_dmap'].values()))).reshape(-1)
            fac_dmap_flatten = np.concatenate(unnestedl2(list(res[id]['fac_dmap'].values()))).reshape(-1)
            edg_map_flatten = np.concatenate(list(res[id]['edg_map'])).reshape(-1)
            
            # record dims
            if self.dims == None:
                self.dims = dict()
                self.dims['featuregb'] = len(featuregb_flatten)
                self.dims['self_vec'] = len(self_vec_flatten)
                self.dims['res1d'] = len(res1d_flatten)
                self.dims['r_dmap'] = len(r_dmap_flatten)
                self.dims['u_dmap'] = len(u_dmap_flatten)
                self.dims['fac_dmap'] = len(fac_dmap_flatten)
                self.dims['edg_map'] = len(edg_map_flatten)
            
            sa_obs = np.concatenate([featuregb_flatten, self_vec_flatten, res1d_flatten, r_dmap_flatten, u_dmap_flatten, fac_dmap_flatten, edg_map_flatten])
            assert len(sa_obs) == O_DIM
            obs_list[id] = sa_obs.astype(DTYPE)

        return obs_list

    def to_featureself(self, id, res1d):
        type2digit = {'worker':0, 'cart':1, 'citytile': 2}
        coeff = {'worker':1, 'cart':100/2000, 'citytile': 0}
        fibbo = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        unit_coeff = {'worker': 10/4, 'cart': 10/5, 'citytile': 1}

        # self_vec
        self_vec = {}
        x, y = self.id_info[id]['loc'][self.id_info[id]['team']]
        self_vec['loc'] = np.array([x, y], dtype=DTYPE)
        self_vec['type'] = oh(type2digit[self.id_info[id]['type']], 3)
        self_vec['at_city'] = res1d[id]['at_city'] if 'at_city' in res1d[id].keys() else np.ones(1, dtype=DTYPE) # citytile is always at city
        self_vec['is_alive'] = np.array([self.id_info[id]['is_alive']], dtype=DTYPE)
        self_vec['oh_loc'] = np.concatenate([oh(int(x), 32), oh(int(y), 32)])
        self_vec['oh_cooldown'] = oh(int(res1d[id]['cooldown'] * unit_coeff[self.id_info[id]['type']]), 10) if 'cooldown' in res1d[id].keys() else oh(1, 10)
        self_vec['wood_carry'] = res1d[id]['wood_carry'] if 'wood_carry' in res1d[id].keys() else np.zeros(1, dtype=DTYPE)
        self_vec['coal_carry'] = res1d[id]['coal_carry'] if 'coal_carry' in res1d[id].keys() else np.zeros(1, dtype=DTYPE)
        self_vec['uranium_carry'] = res1d[id]['uranium_carry'] if 'uranium_carry' in res1d[id].keys() else np.zeros(1, dtype=DTYPE)
        self_vec['can_bc'] = np.array([int((self_vec['wood_carry'][0] + self_vec['coal_carry'][0] + self_vec['uranium_carry'][0]) == 100)], dtype=DTYPE)
        self_vec['oh_wood_carry'] = oh(int(self.id_info[id]['wood_carry'] * coeff[self.id_info[id]['type']]), 100+1)
        self_vec['oh_coal_carry'] = oh(int(self.id_info[id]['coal_carry'] * coeff[self.id_info[id]['type']]), 100+1)
        self_vec['oh_uranium_carry'] = oh(int(self.id_info[id]['uranium_carry'] * coeff[self.id_info[id]['type']]), 100+1)
        self_vec['fuel'] = np.array([self.id_info[id]['fuel']], dtype=DTYPE)
        self_vec['oh_fuel'] = oh(int(sum([fib < res1d[id]['fuel'] for fib in fibbo])), 16) if 'fuel' in res1d[id].keys() else oh(0, 16)

        # edg_map
        _edg_map = np.zeros((self.width, self.height), dtype=DTYPE)
        edge_width = int(self.width - self.origin_width)//2
        edge_height = int(self.height - self.origin_height)//2
        _edg_map[edge_width:self.width-edge_width, edge_height:self.height-edge_height] = np.ones((self.origin_width, self.origin_height), dtype=DTYPE)
        edg_map = dupli(_edg_map, 1)
        # raise NameError(self.width, self.origin_width, _edg_map)

        
        # record dims
        for k in self_vec.keys():
            self.self_vec_dims[k] = len(self_vec[k])

        self.edg_map_dims = {
            'loc': W_RF * H_RF * len(edg_map),
        }

        return self_vec, edg_map

    def to_featuregb(self, team, obs):
        globalCityIDCount = self.game_state.players[0].city_tile_count + self.game_state.players[1].city_tile_count
        globalUnitIDCount = len(self.game_state.players[0].units) + len(self.game_state.players[1].units)
        step = int(obs['step'])
        rps = obs['updates'][0:4]


        # prepare arrays
        obs_list_len = np.zeros(1, dtype=DTYPE)
        gcity_count = np.zeros((self.max_citytile_num + 1) * 2, dtype=DTYPE)
        gunit_count = np.zeros((self.max_citytile_num + 1) * 2, dtype=DTYPE)
        #step_count = np.zeros(360, dtype=DTYPE)
        day_count = np.zeros(9, dtype=DTYPE)
        hour_count = np.zeros(40, dtype=DTYPE)
        is_night = np.zeros(1, dtype=DTYPE)
        rp = np.zeros(1, dtype=DTYPE)
        can_coal = np.zeros(1, dtype=DTYPE)
        can_uranium = np.zeros(1, dtype=DTYPE)

        for _rps in rps:
            rp_info = _rps.split(' ')
            # research points
            try:
                assert rp_info[0] == 'rp'
            except:
                continue
            _team = int(rp_info[1])
            
            # write to global_info
            self.global_info['rp'][_team] = int(rp_info[2])
            
            if _team == team:
                rp[0] = rp_info[2]
                can_coal[0] = (rp[0] >= 50)
                can_uranium[0] = (rp[0] >= 200)

        gcity_count[min(globalCityIDCount,len(gcity_count)-1)] = 1
        gunit_count[min(globalUnitIDCount,len(gunit_count)-1)] = 1
        day_count[min(int(step / 40), 8)] = 1 # last step is same with 321
        hour_count[step % 40] = 1

        is_night[0] = (step % 40 >= 30) if step else 0
        
        # write to global_info
        self.global_info['step'] = step
        self.global_info['is_night'] = is_night[0]
        
        # record dims
        self.featuregb_dims = {
            'obs_list_len': len(obs_list_len),
            'city_count': len(gcity_count),
            'unit_count': len(gunit_count),
            'day_count': len(day_count),
            'hour_count': len(hour_count),
            'is_night': len(is_night),
            'rp': len(rp),
            'can_coal': len(can_coal),
            'can_uranium': len(can_uranium),
        }
                        
        # record inital quads for each team
        if step == 0:
            for update in obs['updates']:
                update_info = update.split(' ')
                if update_info[0] == 'u':
                    _team = int(update_info[2])
                    x = int(update_info[4])
                    y = int(update_info[5])
                    self.orgin_quads[_team] = self.get_quad((x, y))

        return [obs_list_len, gcity_count, gunit_count, day_count, hour_count, is_night, rp, can_coal, can_uranium]

    def to_feature(self, team, updates):
        res1d = DefaultDict(dict)
        _map_info = {
            'uids': DefaultDict(set), 
            'ct_0': DefaultDict(lambda: 0), 
            'ct_1': DefaultDict(lambda: 0), 
            'unit_0': DefaultDict(str), 
            'unit_1': DefaultDict(str), 
            'road': DefaultDict(lambda: 0),
            'is_res': DefaultDict(lambda: 0),
            }


        # prepare for 1d
        digit2type = {0: 'worker', 1: 'cart'}

        is_team = np.zeros(1, dtype=DTYPE)
        loc = np.zeros(2, dtype=DTYPE)
        dist = np.array([62,], dtype=DTYPE)
        at_city = np.zeros(1, dtype=DTYPE)
        cooldown = np.zeros(1, dtype=DTYPE)
        wood_carry = np.zeros(1, dtype=DTYPE)
        coal_carry = np.zeros(1, dtype=DTYPE)
        uranium_carry = np.zeros(1, dtype=DTYPE)
        fuel = np.zeros(1, dtype=DTYPE)
        lightupkeep = np.zeros(1, dtype=DTYPE)
        nights = np.zeros(1, dtype=DTYPE)
        hunger = np.zeros(1, dtype=DTYPE)

        # prepare for 2d
        blank = np.zeros((self.width, self.width), dtype=DTYPE)

        # -- {WOOD: [AMOUNT], COAL: [AMOUNT], URANIUM: [AMOUNT]}
        r_dmap = {'wood': dupli(blank, 1), 'coal': dupli(blank, 1), 'uranium': dupli(blank, 1)}
        
        # -- {WORKER: [SELF_LOC, RIVAL_LOC, TOTAL_CARRY], CART: [SELF_LOC, RIVAL_LOC, TOTAL_CARRY]}
        u_dmap = {0: dupli(blank, 3), 1: dupli(blank, 3)} 
        
        # -- {CITY_TILES: [SELF_LOC, RIVAL_LOC, FUEL, LIGHTUPKEEP], ROAD: [LEVEL]}
        fac_dmap = {'ct': dupli(blank, 6), 'ccd':dupli(blank, 1)}

        for update in updates:
            update_info = update.split(' ')

            # resources
            if update_info[0] == 'r':
                res_type = update_info[1]
                x = int(update_info[2])
                y = int(update_info[3])
                x, y = self.coord(x, y)
                if self.is_trans: x, y = self.coord_trans(x, y, team, self.orgin_quads)
                
                amount = int(update_info[4])
                if update_info[1] in r_dmap.keys():
                    r_dmap[res_type][0][x, y] = amount

                # write to map_info
                _map_info['is_res'][x, y] = 1
            
            # units
            if update_info[0] == 'u':
                unit_type = int(update_info[1])
                _team = int(update_info[2])
                pid = self.gen_pid(update_info[3])
                xo = int(update_info[4])
                yo = int(update_info[5])
                x, y = self.coord(xo, yo)
                if self.is_trans: x, y = self.coord_trans(x, y, _team, self.orgin_quads)
                _cooldown = float(update_info[6])
                _wood = int(update_info[7])
                _coal = int(update_info[8])
                _uranium = int(update_info[9])

                # write to 1d array
                is_team[0] = (_team == team)
                loc[0], loc[1] = x, y
                cooldown[0] = _cooldown
                wood_carry[0] = _wood
                coal_carry[0] = _coal
                uranium_carry[0] = _uranium

                # append to res1d
                res1d[pid]['is_team'] = is_team.copy()
                res1d[pid]['loc'] = loc.copy()
                res1d[pid]['dist'] = dist.copy()
                res1d[pid]['at_city'] = at_city.copy()
                res1d[pid]['cooldown'] = cooldown.copy()
                res1d[pid]['wood_carry'] = wood_carry.copy()
                res1d[pid]['coal_carry'] = coal_carry.copy()
                res1d[pid]['uranium_carry'] = uranium_carry.copy()


                # write to 2d array
                if unit_type in u_dmap.keys():
                    if _team == team:
                        u_dmap[unit_type][0][x, y] = 1
                    else:
                        u_dmap[unit_type][1][x, y] = 1
                    u_dmap[unit_type][2][x, y] = _wood + _coal + _uranium
                
                # write to id_info
                self.id_info[pid]['is_alive'] = 1
                self.id_info[pid]['loc'][_team] = [x, y]
                self.id_info[pid]['loc'][1-_team] = self.coord_trans(*self.coord(xo, yo), 1-_team, self.orgin_quads) if self.is_trans else [x, y]
                self.id_info[pid]['type'] = digit2type[unit_type]
                self.id_info[pid]['team'] = _team
                self.id_info[pid]['uid'] = self.PID2UID[pid]
                self.id_info[pid]['wood_carry'] = _wood
                self.id_info[pid]['coal_carry'] = _coal
                self.id_info[pid]['uranium_carry'] = _uranium
                self.id_info[pid]['cooldown'] = _cooldown

                # write to map_info
                _map_info['uids'][x, y].add(self.PID2UID[pid])
                if _team == 0: _map_info['unit_0'][x, y] = digit2type[unit_type]
                if _team == 1: _map_info['unit_1'][x, y] = digit2type[unit_type]
                
            
            # facilities
            if update_info[0] == 'c':
                _team = int(update_info[1])
                city_id = update_info[2]
                _fuel = int(update_info[3])
                _lightupkeep = int(update_info[4])

                # write to city_info
                self.city_info[city_id]['fuel'] = _fuel
                self.city_info[city_id]['lightupkeep'] = _lightupkeep
                self.city_info[city_id]['tct_count'] = self.get_tct_count(_team, city_id)
                
            if update_info[0] == 'ct':
                _team = int(update_info[1])
                city_id = update_info[2]
                pid = self.gen_pid('_'.join([update_info[2], update_info[3], update_info[4]]))
                xo = int(update_info[3])
                yo = int(update_info[4])
                x, y = self.coord(xo, yo)
                if self.is_trans: x, y = self.coord_trans(x, y, _team, self.orgin_quads)
                _cooldown = float(update_info[5])
                

                # write to 1d array
                # team
                is_team[0] = (_team == team)
                loc[0], loc[1] = x, y
                cooldown[0] = _cooldown
                # fuel[0] = _fuel
                # lightupkeep[0] = _lightupkeep

                # append to res1d
                res1d[pid]['is_team'] = is_team.copy()
                res1d[pid]['loc'] = loc.copy()
                res1d[pid]['dist'] = dist.copy()
                res1d[pid]['cooldown'] = cooldown.copy()
                res1d[pid]['fuel'] = fuel.copy()
                res1d[pid]['lightupkeep'] = lightupkeep.copy()
                res1d[pid]['nights'] = nights.copy()
                res1d[pid]['hunger'] = hunger.copy()

                # write to 2d array
                if _team == team:
                    fac_dmap['ct'][0][x, y] = 1
                else:
                    fac_dmap['ct'][1][x, y] = 1
                

                # write to id_info
                self.id_info[pid]['is_alive'] = 1
                self.id_info[pid]['loc'][_team] = [x, y]
                self.id_info[pid]['loc'][1-_team] = self.coord_trans(*self.coord(xo, yo), 1-_team, self.orgin_quads) if self.is_trans else [x, y]
                self.id_info[pid]['type'] = 'citytile'
                self.id_info[pid]['team'] = _team
                self.id_info[pid]['uid'] = self.PID2UID[pid]
                self.id_info[pid]['cooldown'] = _cooldown

                # write to map_info
                _map_info['uids'][x, y].add(self.PID2UID[pid])
                if _team == 0: _map_info['ct_0'][x, y] = 1
                if _team == 1: _map_info['ct_1'][x, y] = 1

            if update_info[0] == 'ccd':
                x = int(update_info[1])
                y = int(update_info[2])
                x, y = self.coord(x, y)
                if self.is_trans: x, y = self.coord_trans(x, y, team, self.orgin_quads)
                level = float(update_info[3])
                fac_dmap['ccd'][0][x, y] = level
                
                # write to map_info
                _map_info['road'][x, y] = level

        # config res1d[pid]['at_city']
        for pid in res1d.keys():
            if self.id_info[pid]['type'] != 'citytile':
                _team = self.id_info[pid]['team']
                x, y = self.id_info[pid]['loc'][_team]
                _at_city = (_map_info['ct_0'][x, y] or _map_info['ct_1'][x, y])
                res1d[pid]['at_city'][0] = _at_city
                self.id_info[pid]['at_city'] = [uid for uid in _map_info['uids'][x, y] if uid[0] == 'c'][0] if _at_city else 0
            else:
                self.id_info[pid]['at_city'] = self.PID2UID[pid]
                
        # compute ct_count for each team and attrs for citytile
        ct_count = {0: 0, 1: 0}
        unit_count = {0: 0, 1: 0}
        for pid in res1d.keys():
            _team = self.id_info[pid]['team']
            
            if self.id_info[pid]['type'] == 'citytile':
                # for ct_count
                ct_count[_team] += 1
                
                # for lightupkeep
                x, y = self.id_info[pid]['loc'][_team]
                _map = _map_info['ct_'+str(_team)]
                adj_city_num = sum([_map[x+1, y], _map[x-1, y], _map[x, y+1], _map[x, y-1]])
                _lightupkeep = int(23 - 5 * adj_city_num)
                res1d[pid]['lightupkeep'][0] = _lightupkeep
                self.id_info[pid]['lightupkeep'] = _lightupkeep
                fac_dmap['ct'][3][x, y] = _lightupkeep

                # for fuel
                city_id = '_'.join(self.id_info[pid]['uid'].split('_')[0:2])
                _fuel = float(self.city_info[city_id]['fuel'] * _lightupkeep / self.city_info[city_id]['lightupkeep'])
                res1d[pid]['fuel'][0] = _fuel
                self.id_info[pid]['fuel'] = _fuel
                fac_dmap['ct'][2][x, y] = _fuel

                # for nights
                night_hour_left = 10 if (self.global_info['step'] % 40) < 30 else 40 - self.global_info['step'] % 40
                _nights = int(bool(_fuel // (_lightupkeep * night_hour_left)))
                res1d[pid]['nights'][0] = _nights
                self.id_info[pid]['nights'] = _nights
                fac_dmap['ct'][4][x, y] = _nights

                # for hunger
                _hunger = max(0, night_hour_left * _lightupkeep - _fuel + 1)
                res1d[pid]['hunger'][0] = _hunger
                self.id_info[pid]['hunger'] = _hunger
                fac_dmap['ct'][5][x, y] = _hunger

            else:
                unit_count[_team] += 1


        # write to global_info
        self.global_info['ct_count'] = ct_count
        self.global_info['unit_count'] = unit_count

        # record dims
        for key in r_dmap.keys():
            self.r_dmap_dims[key] = W_RF * H_RF * len(r_dmap[key])
        for key in u_dmap.keys():
            self.u_dmap_dims[key] = W_RF * H_RF * len(u_dmap[key])
        for key in fac_dmap.keys():
            self.fac_dmap_dims[key] = W_RF * H_RF * len(fac_dmap[key])
        
        return res1d, r_dmap, u_dmap, fac_dmap, _map_info

    def pack_res1d(self, res1d):
        self_ut_ids = []
        self_ct_ids = []
        op_ut_ids = []
        op_ct_ids = []
        
        # cluster ids by type
        for pid in res1d.keys():
            if self.id_info[pid]['team']:
                if self.id_info[pid]['type'] == 'citytile':
                    self_ct_ids.append(pid)
                else:
                    self_ut_ids.append(pid)
            else:
                if self.id_info[pid]['type'] == 'citytile':
                    op_ct_ids.append(pid)
                else:
                    op_ut_ids.append(pid)

        self_ut_ids.sort(key=lambda pid: res1d[pid]['dist'][0])
        self_ct_ids.sort(key=lambda pid: res1d[pid]['dist'][0])
        op_ut_ids.sort(key=lambda pid: res1d[pid]['dist'][0])
        op_ct_ids.sort(key=lambda pid: res1d[pid]['dist'][0])
        
        # 10 Nearest Neighbour
        self_ut_vecs_10nn = [np.concatenate(list(res1d[pid].values())) for pid in self_ut_ids[0:10]]
        self_ct_vecs_10nn = [np.concatenate(list(res1d[pid].values())) for pid in self_ct_ids[0:10]]
        op_ut_vecs_10nn = [np.concatenate(list(res1d[pid].values())) for pid in op_ut_ids[0:10]]
        op_ct_vecs_10nn = [np.concatenate(list(res1d[pid].values())) for pid in op_ct_ids[0:10]]

        # rest random shuffle
        self_ut_ids_rest = copy.deepcopy(self_ut_ids[10:160])
        self_ct_ids_rest = copy.deepcopy(self_ct_ids[10:160])
        op_ut_ids_rest = copy.deepcopy(self_ct_ids[10:160])
        op_ct_ids_rest = copy.deepcopy(op_ct_ids[10:160])

        random.shuffle(self_ut_ids_rest)
        random.shuffle(self_ct_ids_rest)
        random.shuffle(op_ut_ids_rest)
        random.shuffle(op_ct_ids_rest)

        self_ut_vecs_rest = [np.concatenate(list(res1d[pid].values())) for pid in self_ut_ids_rest]
        self_ct_vecs_rest = [np.concatenate(list(res1d[pid].values())) for pid in self_ct_ids_rest]
        op_ut_vecs_rest = [np.concatenate(list(res1d[pid].values())) for pid in op_ut_ids_rest]
        op_ct_vecs_rest = [np.concatenate(list(res1d[pid].values())) for pid in op_ct_ids_rest]

        self_ut_vecs = self_ut_vecs_10nn + self_ut_vecs_rest
        self_ct_vecs = self_ct_vecs_10nn + self_ct_vecs_rest
        op_ut_vecs = op_ut_vecs_10nn + op_ut_vecs_rest
        op_ct_vecs = op_ct_vecs_10nn + op_ct_vecs_rest
        
        # pack to 160 d
        self_ut_vecs += [np.zeros(U_DIM, dtype=DTYPE)] * (self.max_citytile_num - len(self_ut_vecs))
        self_ct_vecs += [np.zeros(CT_DIM, dtype=DTYPE)] * (self.max_citytile_num - len(self_ct_vecs))

        op_ut_vecs += [np.zeros(U_DIM, dtype=DTYPE)] * (self.max_citytile_num - len(op_ut_vecs))
        op_ct_vecs += [np.zeros(CT_DIM, dtype=DTYPE)] * (self.max_citytile_num - len(op_ct_vecs))

        self.res1d_dims = {
            'self_ut_vecs': len(self_ut_vecs) * U_DIM,
            'self_ct_vecs': len(self_ct_vecs) * CT_DIM,
            'op_ut_vecs': len(op_ut_vecs) * U_DIM,
            'op_ct_vecs': len(op_ct_vecs) * CT_DIM,
        }

        return self_ut_vecs + self_ct_vecs + op_ut_vecs + op_ct_vecs

    def show(self, _obs_list):
        digit2type = {0: 'worker', 1: 'cart'}
        def divide_by_dict(array, dim_dict):
            array_diveded = {}
            offset = 0
            for var in dim_dict.keys():
                dim = dim_dict[var]
                array_diveded[var] = array[offset:offset+dim]
                offset += dim
            return array_diveded

        _obs_list_divide = divide_by_dict(_obs_list, self.dims)
        
        featuregb = _obs_list_divide['featuregb']
        self_vec = _obs_list_divide['self_vec']
        res1d = _obs_list_divide['res1d']
        r_dmap = _obs_list_divide['r_dmap']
        u_dmap = _obs_list_divide['u_dmap']
        fac_dmap = _obs_list_divide['fac_dmap']
        edg_map = _obs_list_divide['edg_map']
        

        featuregb_divide = divide_by_dict(featuregb, self.featuregb_dims)
        self_vec_divide = divide_by_dict(self_vec, self.self_vec_dims)
        res1d_divide = divide_by_dict(res1d, self.res1d_dims)
        r_dmap_divide = divide_by_dict(r_dmap, self.r_dmap_dims)
        u_dmap_divide = divide_by_dict(u_dmap, self.u_dmap_dims)
        fac_dmap_divide = divide_by_dict(fac_dmap, self.fac_dmap_dims)
        edg_map_divide = divide_by_dict(edg_map, self.edg_map_dims)

        
        print('----------------------featuregb_divide-------------------')
        pprint(featuregb_divide)
        print('----------------------self_vec_divide--------------------')
        pprint(self_vec_divide)
        print('------------------------res1d_divide---------------------')
        for var in res1d_divide.keys():
            lenth = U_DIM if 'unit' in var else CT_DIM
            res1d2p = res1d_divide[var][0:lenth]
            print(var, res1d2p.dtype, res1d2p.shape)
            print(var, res1d2p)

        print('-----------------------r_dmap_divide---------------------')
        for var in r_dmap_divide.keys():
            print(var, 'amount')
            r_dmap2p = r_dmap_divide[var].reshape(W_RF, H_RF)
            print(r_dmap2p.dtype, r_dmap2p.shape)
            print(r_dmap2p)

        print('-----------------------u_dmap_divide---------------------')
        for var in u_dmap_divide.keys():
            for i, title in enumerate(['self_loc', 'op_loc', 'total_carry']):
                print(digit2type[var], title)
                u_dmap2p = u_dmap_divide[var][i*W_RF*H_RF:(i+1)*W_RF*H_RF].reshape(W_RF, H_RF)
                print(u_dmap2p.dtype, u_dmap2p.shape)
                print(u_dmap2p)

        print('----------------------fac_dmap_divide--------------------')
        for i, title in enumerate(['self_loc', 'op_loc', 'fuel', 'lightupkeep', 'nights', 'hunger']):
            print('citytile', title)
            fac_dmap2p = fac_dmap_divide['ct'][i*W_RF*H_RF:(i+1)*W_RF*H_RF].reshape(W_RF, H_RF)
            print(fac_dmap2p.dtype, fac_dmap2p.shape)
            print(fac_dmap2p)
        print('road')
        fac_dmap2p = fac_dmap_divide['ccd'].reshape(W_RF, H_RF)
        print(fac_dmap2p.dtype, fac_dmap2p.shape)
        print(fac_dmap2p)
        
        print('----------------------edg_map_divide--------------------')
        for i, title in enumerate(self.edg_map_dims.keys()):
            print('edg_map', title)
            edg_map = edg_map_divide[title].reshape(W_RF, H_RF)
            print(edg_map.dtype, edg_map.shape)
            print(edg_map)



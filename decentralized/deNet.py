import torch
from torch import nn
import torch.nn.functional as F

class DeNet(nn.Module):
    def __init__(self, feature_dims, actions_dims):
        super(DeNet, self).__init__()

        global_dim, self_dim, unit_dim, city_dim, imagelike_dim, self.unit_single_dim, self.city_single_dim, self.unit_num, self.city_num, self.nearest_num = feature_dims
        self.worker_act_dim, self.city_act_dim = actions_dims        
        self.channel = imagelike_dim[0]

        self.emb_size = 64
        self.entity = 6

        self.rest = self.unit_num - self.nearest_num
        
        self.self_fc1 = nn.Linear(self_dim, 128)
        self.self_fc2 = nn.Linear(128, self.emb_size)

        self.global_fc1 = nn.Linear(global_dim, 128)
        self.global_fc2 = nn.Linear(128, self.emb_size)

        self.unit_fc1 = nn.Linear(self.unit_single_dim, 128)
        self.unit_fc2 = nn.Linear(128, self.emb_size)
        self.city_fc1 = nn.Linear(self.city_single_dim, 128)
        self.city_fc2 = nn.Linear(128, self.emb_size)

        self.opunit_fc1 = nn.Linear(self.unit_single_dim, 128)
        self.opunit_fc2 = nn.Linear(128, self.emb_size)
        self.opcity_fc1 = nn.Linear(self.city_single_dim, 128)
        self.opcity_fc2 = nn.Linear(128, self.emb_size)

        self.conv1 = nn.Conv2d(self.channel, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.image_fc3 = nn.Linear(4 * 4 * 16, 128)

        self.all_fc4 = nn.Linear(128 + self.emb_size * self.entity, 512) 
        self.all_fc5 = nn.Linear(512, 256)

        # actor params
        self.worker_act = nn.Linear(256, self.worker_act_dim)
        self.city_act = nn.Linear(256, self.city_act_dim)
        # critic params
        self.critic = nn.Linear(256, 1)


    def forward(self, x):

        global_feature, self_feature, unit_feature, city_feature, opunit_feature, opcity_feature, imagelike_feature = x

        # self feature forward embedding
        self_o = F.relu(self.self_fc2(F.relu(self.self_fc1(self_feature))))
        global_o = F.relu(self.global_fc2(F.relu(self.global_fc1(global_feature))))
        # self team unit forward embedding
        unit_emb = F.relu(self.unit_fc2(F.relu(self.unit_fc1(unit_feature)))) # batchsize*unit_num, unit_single_dim
        unit_o, _ = torch.max(unit_emb, dim = 1)
        # self team city forward embedding
        city_emb = F.relu(self.city_fc2(F.relu(self.city_fc1(city_feature)))) # batchsize * unit_num, unit_single_dim
        city_o, _ = torch.max(city_emb, dim = 1)
        # opponent team unit forward embedding
        opunit_emb = F.relu(self.opunit_fc2(F.relu(self.opunit_fc1(opunit_feature)))) # batchsize*unit_num, unit_single_dim
        opunit_o, _ = torch.max(opunit_emb, dim = 1)
        # opponent team city forward embedding
        opcity_emb = F.relu(self.opcity_fc2(F.relu(self.opcity_fc1(opcity_feature)))) # batchsize * unit_num, unit_single_dim
        opcity_o, _ = torch.max(opcity_emb, dim = 1)

        # image like feature 
        image_x = F.relu(self.conv1(imagelike_feature))
        image_x = F.relu(self.conv2(image_x))
        image_x = F.relu(self.conv3(image_x))
        image_x = image_x.view(image_x.size(0), -1)
        image_o = F.relu(self.image_fc3(image_x))
        # final fc forward

        o = torch.cat([image_o, global_o, self_o, city_o, unit_o, opcity_o, opunit_o], 1)        
        o = F.relu(self.all_fc5(F.relu(self.all_fc4(o))))

        # actor forward
        worker_action = self.worker_act(o)

        return worker_action
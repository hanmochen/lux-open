import torch
from torch import nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, padding=2):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(),
        )
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        # self.leaky_relu = nn.LeakyReLU()
        self.shortcut = nn.Sequential()
        self.selayer = SELayer(out_channel)
        self._init_w_b()

    def forward(self, x):
        out = self.left(x)
        out = self.selayer(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

    def _init_w_b(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _init_w_b(layers):
    for layer in layers:
        nn.init.kaiming_normal_(layer.weight)
        #nn.init.zeros_(layer.bias)

class Net(nn.Module):
    def __init__(self, model_param, global_feature_dims, map_channel, map_size, n_actions):
        super(Net, self).__init__()

        # print(model_param)
        # print(model_param['emb_dim'])

        emb_dim = int(model_param["emb_dim"])
        n_res_blocks = model_param["n_res_blocks"]
        global_channel = model_param["global_channel"]
        all_channel = model_param["all_channel"]
        self.all_channel = all_channel


        self.global_feature_dims = global_feature_dims
        self.map_channels = map_channel
        self.map_size = map_size
        self.n_actions = n_actions
        self.embedding_layer = nn.Linear(global_feature_dims[0], emb_dim)

        global_channel = global_feature_dims[1]+emb_dim
        input_channel = global_channel + map_channel # 64

        self.res_blocks = nn.ModuleList([
            ResidualBlock(all_channel, all_channel) for _ in range(n_res_blocks)
            ])

        self.spectral_norm = nn.utils.spectral_norm(nn.Conv2d(all_channel,all_channel,kernel_size=1, stride=1, padding=0, bias=False))

        self.conv1 = nn.Conv2d(emb_dim, emb_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(global_feature_dims[1], global_feature_dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(global_channel, global_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.input_conv1 = nn.Conv2d(input_channel, all_channel, kernel_size=1, stride=1, padding=0, bias=False)  
        self.worker_action = nn.Conv2d(all_channel, n_actions[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.cart_action = nn.Conv2d(all_channel, n_actions[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.citytile_action = nn.Conv2d(all_channel, n_actions[2], kernel_size=1, stride=1, padding=0, bias=False)
        _init_w_b([self.worker_action, self.cart_action, self.citytile_action])
        self.critic_fc = nn.Linear(all_channel, 1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.input_conv1.weight)
        nn.init.xavier_normal_(self.critic_fc.weight)

    def forward(self, x):
        global_emb_feature, global_no_emb_feature, map_feature = x 
        global_emb_feature = self.embedding_layer(global_emb_feature)
        global_emb_feature = global_emb_feature.view(-1,global_emb_feature.shape[1],1,1).expand(-1,global_emb_feature.shape[1],self.map_size,self.map_size)
        global_emb_feature = self.conv1(global_emb_feature)
        global_emb_feature = F.leaky_relu(global_emb_feature)
        global_no_emb_feature = global_no_emb_feature.view(-1,global_no_emb_feature.shape[1],1,1).expand(-1,global_no_emb_feature.shape[1],self.map_size,self.map_size)
        global_no_emb_feature = self.conv2(global_no_emb_feature)
        global_no_emb_feature = F.leaky_relu(global_no_emb_feature)

        global_feature = torch.cat([global_emb_feature, global_no_emb_feature], dim=1)
        global_feature = self.conv3(global_feature)

        x = torch.cat([global_feature, map_feature], dim=1)

        x = self.input_conv1(x)

        for block in self.res_blocks:
            x = block(x)
        
        x = self.spectral_norm(x)

        worker_action = self.worker_action(x)
        worker_action = worker_action.flatten(start_dim=2,end_dim=3)
        worker_action = worker_action.transpose(1,2)

        cart_action = self.cart_action(x)
        cart_action = cart_action.flatten(start_dim=2,end_dim=3)
        cart_action = cart_action.transpose(1,2)

        citytile_action = self.citytile_action(x)
        citytile_action = citytile_action.flatten(start_dim=2,end_dim=3)
        citytile_action = citytile_action.transpose(1,2)

        x = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1)/(self.map_size * self.map_size)

        critic_value = self.critic_fc(x.view(-1, self.all_channel)).view(-1)
        # critic_value = F.tanh(self.critic_fc(x.view(-1, self.all_channel))).view(-1)
        return [worker_action,cart_action,citytile_action], critic_value

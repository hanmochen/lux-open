import argparse
import importlib
from datetime import datetime
import os
import json
import shutil
from env.lux_env import ProxyEnv, Observation
from agent import agent as rlAgent
from decentralized.agent import agent as deAgent
from opponent.lux_ai.rl_agent.rl_agent import agent as opAgent
model_dir = "./models/"

class EvalEnv():
    def __init__(self, args):
        self.args = args
        self.mode = args.mode # "op" "self" "old" "normal"
        if self.mode == "op":
            self.agent0 = self.get_agent(self.args.model0)
            self.agent1 = opAgent
            self.name0 = self.args.model0
            self.name1 = "op"
        elif self.mode == "old":
            self.agent0 = self.get_agent(self.args.model0)
            self.agent1 = deAgent
            self.name0 = self.args.model0
            self.name1 = "old"
        elif self.mode == "self":
            self.args.model1 = self.args.model0 if self.args.model1 is None else self.args.model1
            self.agent0 = self.get_agent(self.args.model0)
            self.agent1 = self.get_agent(self.args.model1)
            self.name0 = self.args.model0
            self.name1 = self.args.model1 
        else:
            print("Invalid mode")

    def get_agent(self, model):
        rlAgent.config['model_path'] = os.path.join(model_dir, model, "model.pt")
        rlAgent.config['map_size'] = self.args.map_size
        return rlAgent.Agent(rlAgent.config).get_actions

    def run(self):
        # store the replay.json
        replay_path = self.mode + "_" + datetime.now().strftime("%m%d-%H%M")
        replay_path = os.path.join("replays", replay_path)
        agent0_win_path = os.path.join(replay_path, self.name0)
        agent1_win_path = os.path.join(replay_path, self.name1)
        if not os.path.exists(replay_path):
            os.makedirs(replay_path)
            os.makedirs(agent0_win_path)
            os.makedirs(agent1_win_path)

        for i in range(self.args.num_games):
            print('Game {}'.format(i))

            env = ProxyEnv(map_size=self.args.map_size)
            conf = env.conf
            obs, _, done = env.reset()
            step = 0
            while not done:
                observation = Observation()
                observation["updates"] = obs[0]["updates"]
                observation["remainingOverageTime"] = 60.
                observation["step"] = step
                actions = [[],[]]
                observation.player = 0
                actions[0] = self.agent0(observation,conf)
                observation.player = 1
                actions[1] = self.agent1(observation,conf)
                obs, _, done = env.step(actions)
                step += 1
                print("Step: {}".format(step))
            env.save(replay_path+'/replay_{}.json'.format(i))
            with open(replay_path+'/replay_{}.json'.format(i)) as f:
                game_result = json.load(f)
                game_rewards = game_result["rewards"]
                if game_rewards[0] > game_rewards[1]:
                    print("    Agent0 win")
                elif game_rewards[0] < game_rewards[1]:
                    print("    Agent1 win")
                else:
                    print("    Draw")

        # organize the game result
        total_num = 0
        agent0_win_num = 0
        agent1_win_num = 0
        agent0_win_list = []
        agent1_win_list = []
        folder_dir = replay_path
        for file_dir in os.listdir(folder_dir):
            file_dir = os.path.join(folder_dir, file_dir)
            if file_dir.endswith("json"):
                total_num += 1
                with open(file_dir, "r") as f:
                    game_result = json.load(f)
                game_rewards = game_result["rewards"]
                if game_rewards[0] > game_rewards[1]:
                    agent_win = 0
                else:
                    agent_win = 1
                if agent_win == 0:
                    agent0_win_num += 1
                    agent0_win_list.append(file_dir.split("/")[-1])
                    shutil.move(file_dir, agent0_win_path)
                else:
                    agent1_win_num += 1
                    agent1_win_list.append(file_dir.split("/")[-1])
                    shutil.move(file_dir, agent1_win_path)
        print()
        print("----------- RESULTS -----------")
        print("agent0 - {}".format(self.name0))
        print("agent1 - {}".format(self.name1))
        print()
        print("agent 0 win {} games over {} games".format(agent0_win_num, total_num))
        # print("agent 0 win in the following games:")
        # for fi in sorted(agent0_win_list):
        #     print("        " + fi)
        # print()
        print("agent 1 win {} games over {} games".format(agent1_win_num, total_num))
        # print("agent 1 win in the following games:")
        # for fi in sorted(agent1_win_list):
        #     print("        " + fi)
        return None
                
        


parser = argparse.ArgumentParser(description='Lux AI Evaluation')
parser.add_argument('--mode','-m',help='Evaluation Mode',default='op')
parser.add_argument('--model0','-p0',help='The model path of agent0 (Optional)',default="best12")
parser.add_argument('--model1','-p1',help='The model path of agent1 (Optional)',default=None)
parser.add_argument('--num_games', '-n', help='the number of games', default=1, type=int)
parser.add_argument('--map_size', '-s', help='the size of map', default=12,type=int)
parser.add_argument('--seed', '-r', help='the random seed', default=42, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    eval_env = EvalEnv(args)
    eval_env.run()
    print("Done")





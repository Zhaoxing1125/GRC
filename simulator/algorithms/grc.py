import numpy as np
import random
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GlobalAttention
import torch.nn.functional as F
from reward_model import *
from env_model import *

sys.path.append('../')
from objects import *

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    
if __name__=="__main__":
    def get_dataset(date_idx):
        order_num_data = pickle.load(open('../result/stats_active_orders'+str(date_idx)+'_2kd_05o_mdp', 'rb'))
        driver_num_data = pickle.load(open('../result/stats_active_drivers'+str(date_idx)+'_2kd_05o_mdp', 'rb'))

        dataset = {}
        for time in driver_num_data.keys():
            valid_gids = set(list(driver_num_data[time].keys()) + list(order_num_data[time].keys()) )

            dataset[time] = {}
            for gid in valid_gids:
                data = []

                if gid in driver_num_data[time]:
                    d_num = len(driver_num_data[time][gid])
                else:
                    d_num = 0
                if gid in order_num_data[time]:
                    o_num = len(order_num_data[time][gid])
                else:
                    o_num = 0
                data.append(d_num - o_num)

                if gid in order_num_data[time]:
                    ov_list = [v[-1] for v in order_num_data[time][gid].values()]
                    ovp_list = [v[-1]/(v[-2] - v[-4]) for v in order_num_data[time][gid].values()]
                    data.extend([
                        np.mean(ov_list),
                        np.min(ov_list),
                        np.max(ov_list),
                        np.mean(ovp_list)]
                    )
                else:
                    data.extend([
                        0.0,
                        0.0,
                        0.0,
                        0.0]
                    )

                dataset[time][gid] = data
            
            for gid in range(963):
                if gid not in valid_gids:
                    dataset[time][gid] = [0.0, 0.0, 0.0, 0.0, 0.0]

            myKeys = list(dataset[time].keys())
            myKeys.sort()
            sorted_dict = {i: dataset[time][i] for i in myKeys}
            dataset[time] = sorted_dict

        return dataset
    
    env_model = ENVM(5, 5)
    env_model.load_state_dict(torch.load('./result/env_model_1'))
    reward_model = RM(5, 1)
    reward_model.load_state_dict(torch.load('./result/reward_model_1'))

    model = Policy(5 + 18, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    edge_index = pickle.load(open('./result/edge_index', 'rb'))
    mdp_table = pickle.load(open('./result/mdp_table_2kd_05o_3', 'rb'))
    criterion = nn.SmoothL1Loss()


    # initialize
    for eps in range(100):
        for date_idx in [4, 5, 6]:
            dataset = get_dataset(date_idx)
            loss = 0
            for time in dataset.keys():

                time_step = [0.0] * len(dataset.keys())
                time_step[time - 7*6] = 1.0
                state = torch.tensor([x for x in dataset[time].values()]).float()
                rw1 = reward_model(state).detach()

                _state = torch.concat((state, torch.tensor(time_step).float().reshape(1, -1).repeat(len(state), 1)), dim=1)
                pred_action = model(_state)
                action = []
                for gid in range(963):
                    key = (gid, time)
                    if key not in mdp_table:
                        action.append(0)
                    else:
                        action.append(mdp_table[key])
                print(pred_action[:100])
                print(action[:100])
                
                for i in range(len(action)):
                    loss += (action[i] > 0) * (criterion(torch.tensor(action[i]).float(), pred_action[i]) )

            loss /= len(dataset)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Loss:", loss.item())

    torch.save(model.state_dict(), './result/grc_model_middle_1')

    # GRC update
    for eps in range(100):
        for date_idx in [4, 5, 6]:
            dataset = get_dataset(date_idx)
            loss = 0
            for _ in range(10):
                time = np.random.choice(list(dataset.keys()))

                state = torch.tensor([x for x in dataset[time].values()]).float()
                rw1 = reward_model(state).detach()


                time_step = [0.0] * len(dataset.keys())
                time_step[time - 7*6] = 1.0
                _state = torch.concat((state, torch.tensor(time_step).float().reshape(1, -1).repeat(len(state), 1)), dim=1)
                
                pred_action = model(_state)
                action = []
                for gid in range(963):
                    key = (gid, time)
                    if key not in mdp_table:
                        action.append(0)
                    else:
                        action.append(mdp_table[key])
                print(pred_action[:100])
                print(action[:100])
                
                goal_next_state, ord_next_state = env_model.sample(state, rw1.reshape(-1, 1), torch.tensor(pred_action).reshape(-1, 1), edge_index, 20)

                loss += criterion(goal_next_state.detach(), ord_next_state)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Loss:", loss.item())

    torch.save(model.state_dict(), './result/grc_model_1')

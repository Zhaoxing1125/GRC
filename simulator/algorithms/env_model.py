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

sys.path.append('../')
from objects import *

def get_agg_score(rw1, edge_index, layer):
    rw_res = rw1.clone().detach()

    for lay in range(layer):
        _rw_res = []
        for node in range(len(rw1)):
            ad, ad_num = rw_res[node], 0
            for x in edge_index:
                if x[0] == node:
                    ad += rw_res[x[1]]
                    ad_num += 1
            _rw_res.append(ad / ad_num)
        rw_res = torch.stack(_rw_res).reshape(-1)

    return rw_res

class ENVM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ENVM, self).__init__()

        self.hidden = 128

        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
        )

        self.hv = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden * 2),
            nn.ReLU(),
        )

        self.gcn1 = GCNConv(self.hidden + 2, self.hidden * 2)
        self.gcn2 = GCNConv(self.hidden, output_dim + 1)

    def forward(self, x, rw1, action, edge_index):
        edge_index = torch.tensor(edge_index).reshape(2, -1).long()

        x = self.net(x)
        x3 = F.relu(self.gcn1(torch.concat((x, rw1, action), dim=1), edge_index))

        x3 = F.relu(self.hv(x3))

        mean, logvar = x3[:, :self.hidden], x3[:, self.hidden:]

        hidden = mean + torch.randn_like(logvar) * torch.exp(torch.clamp(logvar, min=-5, max=5))

        _x1 = F.relu(self.gcn2(hidden, edge_index))
        _x1, _rw1 = _x1[:, :-1], _x1[:, -1]

        return _x1, _rw1
    
    def sample(self, x, rw1, action, edge_index, Sample_Size):
        edge_index = torch.tensor(edge_index).reshape(2, -1).long()

        x = self.net(x)
        x3 = F.relu(self.gcn1(torch.concat((x, rw1, action), dim=1), edge_index))

        x3 = F.relu(self.hv(x3))

        mean, logvar = x3[:, :self.hidden], x3[:, self.hidden:]

        mean, logvar = mean.repeat(Sample_Size, 1, 1), logvar.repeat(Sample_Size, 1, 1)

        hidden = mean + torch.randn_like(logvar) * torch.exp(torch.clamp(logvar, min=-5, max=5))

        hidden = hidden.reshape(Sample_Size, -1, self.hidden)

        _x1 = F.relu(self.gcn2(hidden, edge_index))
        _x1, _rw1 = _x1[:, :, :-1], _x1[:, :, -1]

        _rw1 = _rw1.mean(1).reshape(Sample_Size)
        _, indices = torch.topk(_rw1, 5)

        all_indices = torch.arange(Sample_Size)
        remaining_indices = all_indices[~torch.isin(all_indices, indices)]

        return _x1[indices, :, :].repeat(3, 1, 1), _x1[remaining_indices, :, :]
    
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

    
    model = ENVM(5, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    reward_model = RM(5, 1)
    reward_model.load_state_dict(torch.load('./result/reward_model_1'))
    edge_index = pickle.load(open('./result/edge_index', 'rb'))
    mdp_table = pickle.load(open('./result/mdp_table_2kd_05o_3', 'rb'))
    criterion = nn.SmoothL1Loss()

    for eps in range(100):
        for date_idx in [4, 5, 6]:
            dataset = get_dataset(date_idx)
            for time in dataset.keys():
                if time + 1 not in dataset:
                    continue

                state = torch.tensor([x for x in dataset[time].values()]).float()
                rw1 = reward_model(state).detach()

                action = []
                for gid in range(963):
                    key = (gid, time)
                    if key not in mdp_table:
                        action.append(0)
                    else:
                        action.append(mdp_table[key])
                
                pred_next_state, pred_rw1 = model(state, rw1.reshape(-1, 1), torch.tensor(action).reshape(-1, 1), edge_index)
                            
                next_state = torch.tensor([x for x in dataset[time+1].values()]).float()
                rw1 = reward_model(next_state).detach().reshape(-1)

                loss = 0
                for i in range(len(next_state)):
                    loss += next_state[i].sum() * (criterion(next_state[i], pred_next_state[i]) + criterion(rw1[i], pred_rw1[i]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Loss:", loss.item())

    torch.save(model.state_dict(), './result/env_model_1')

def edge_index_generation():
    Grids_Dataset  = pickle.load(open('../../dataset/Grids_Dataset', 'rb'))
    Grids = {}
    for key, value in Grids_Dataset.items():
        Grids[key] = Grid(value[0], value[1])

    edge_index = []
    for gid1, grid1 in Grids.items():
        dis = []
        for gid2, grid2 in Grids.items():
            if gid1 == gid2:
                continue

            dis.append([
                gid2,
                (grid1.center[0] - grid2.center[0])**2 + (grid1.center[1] - grid2.center[1])**2
            ])
        
        dis.sort(key=lambda x: x[1])

        for x in range(6):
            edge_index.append([gid1, dis[x][0]])
            edge_index.append([dis[x][0], gid1])

    pickle.dump(edge_index, open('./result/edge_index', 'wb+'))
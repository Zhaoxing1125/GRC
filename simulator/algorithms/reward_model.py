import numpy as np
import random
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class RM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RM, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    

def train(dataset, model, optimizer):
    for time, items in dataset.items():
        for _ in range(100):
            loss, BZ = 0, 64
            for _ in range(BZ):
                sampled_index = np.random.choice(len(items), size=2)
                item1, item2 = items[sampled_index[0]], items[sampled_index[1]]

                g1, g2 = 0, 0
                # d-s gap
                if np.abs(item1[0]) < np.abs(item2[0]):
                    g1 += 1
                else:
                    g2 += 1
                # order avg value
                if item1[1] > item2[1]:
                    g1 += 1
                else:
                    g2 += 1

                # order min value
                if item1[2] > item2[2]:
                    g1 += 1
                else:
                    g2 += 1

                # order max value
                if item1[3] > item2[3]:
                    g1 += 1
                else:
                    g2 += 1

                # order mean value per second
                if item1[4] > item2[4]:
                    g1 += 1
                else:
                    g2 += 1

                score1, score2 = model(torch.tensor(item1).float()), model(torch.tensor(item2).float())
                loss +=  score1 * (g2 - g1) + score2 * (g1 - g2)

            loss /= BZ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())

if __name__ == "__main__":
    def get_dataset(date_idx):
        order_num_data = pickle.load(open('../result/stats_active_orders'+str(date_idx)+'_2kd_05o_mdp', 'rb'))
        driver_num_data = pickle.load(open('../result/stats_active_drivers'+str(date_idx)+'_2kd_05o_mdp', 'rb'))

        dataset = {}
        for time in driver_num_data.keys():
            valid_gids = set(list(driver_num_data[time].keys()) + list(order_num_data[time].keys()) )

            dataset[time] = []
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

                dataset[time].append(data)
        return dataset

    
    model = RM(5, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    for date_idx in [4, 5, 6]:
        train(get_dataset(date_idx), model, optimizer)

    torch.save(model.state_dict(), './result/reward_model_1')
            
            

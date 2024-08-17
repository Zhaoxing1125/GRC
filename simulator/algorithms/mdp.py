import numpy as np
import pickle
import sys
import random
import copy
sys.path.append('../')
from objects import *

np.random.seed(100)
random.seed(100)

Grids_Dataset  = pickle.load(open('../../dataset/Grids_Dataset', 'rb'))
Grids = {}
for key, value in Grids_Dataset.items():
    Grids[key] = Grid(value[0], value[1])

mdp_dict, freq_dict = {}, {}

Order_Data_1101  = pickle.load(open('../result/matched_orders_201611012kd_05o', 'rb'))
Order_Data_1102  = pickle.load(open('../result/matched_orders_201611022kd_05o', 'rb'))
Order_Data_1103  = pickle.load(open('../result/matched_orders_201611032kd_05o', 'rb'))
Order_Dataset = copy.deepcopy(Order_Data_1101)
Order_Dataset.extend(Order_Data_1102)
Order_Dataset.extend(Order_Data_1103)

for time in range(10 * 60 * 60, 7 * 60 * 60, -2):
    begin_time_step = int(time / 60 / 10)
    print(time, begin_time_step)
    for order in Order_Dataset:
        start_grid, start_time, end_grid, end_time, price = order
        if time != start_time:
            continue

        end_time_step = int(end_time / 60 / 10)

        if (end_grid, end_time_step) not in mdp_dict:
            mdp_dict[(end_grid, end_time_step)] = 0
        if (start_grid, begin_time_step) not in mdp_dict:
            mdp_dict[(start_grid, begin_time_step)] = 0
        if (start_grid, begin_time_step) not in freq_dict:
            freq_dict[(start_grid, begin_time_step)] = 0
        freq_dict[(start_grid, begin_time_step)] += 1
        mdp_dict[(start_grid, begin_time_step)] += 1/freq_dict[(start_grid, begin_time_step)] * (price + 0.99 ** (end_time_step - begin_time_step) * mdp_dict[(end_grid, end_time_step)])

pickle.dump(mdp_dict, open('result/mdp_table_2kd_05o_3', 'wb+'))
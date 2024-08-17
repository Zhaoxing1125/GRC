import numpy as np
import pickle
import copy
import sys
import os

sys.path.append('./simulator/')
from objects import *
from datetime import datetime


#read grids
Grids, Grids_Dataset = {}, {}
file = open('./dataset/valid_grid.csv', 'r')
for line in file.readlines():
    text = line.split(',')

    Grids[int(text[0])] = Grid(int(text[0]), [float(text[i]) for i in range(2, len(text))])
    Grids_Dataset[int(text[0])] = [int(text[0]), [float(text[i]) for i in range(2, len(text))]]
file.close()
pickle.dump(Grids_Dataset, open('./dataset/Grids_Dataset', 'wb+'))

#read orders
for data_idx in range(4, 7, 1):
    file = open('./raw_dataset/order_2016110'+str(data_idx), 'r')
    begin_dtime = datetime(2016, 11, data_idx, 0)
    begin_dtimestamp = int(round(begin_dtime.timestamp()))
    order_idx, Orders_Dataset = 0, {}
    for line in file.readlines():
        text = line.split(',')

        start_time = int((int(text[1]) - begin_dtimestamp)/600) #int((int(text[1]) - begin_dtimestamp)/2) * 2
        end_time   = int((int(text[2]) - begin_dtimestamp)/600) #int((int(text[2]) - begin_dtimestamp)/2) * 2
        price = float(text[-1])

        start_gps = [float(text[3]), float(text[4])]
        start_distance = [[gdi, (start_gps[0] - gd.center[0])**2 + (start_gps[1] - gd.center[1])**2 ] for gdi, gd in Grids.items()]
        start_distance.sort(key=lambda x: x[1])
        start_grid = start_distance[0][0]

        end_gps = [float(text[5]), float(text[6])]
        end_distance = [[gdi, (end_gps[0] - gd.center[0])**2 + (end_gps[1] - gd.center[1])**2] for gdi, gd in Grids.items()]
        end_distance.sort(key=lambda x: x[1])
        end_grid = end_distance[0][0]

        if start_time not in Orders_Dataset:
            Orders_Dataset[start_time] = {}
        if start_grid not in Orders_Dataset[start_time]:
            Orders_Dataset[start_time][start_grid] = []
        if [order_idx, start_time, end_time, start_grid, end_grid, price] not in Orders_Dataset[start_time][start_grid]:
            Orders_Dataset[start_time][start_grid].append([order_idx, start_time, end_time, start_grid, end_grid, price])
        

        order_idx += 1

        if order_idx % 100 == 0:
            print(order_idx)

    file.close()
    pickle.dump(Orders_Dataset, open('./dataset/Orders_Dataset_2016110'+str(data_idx), 'wb+'))
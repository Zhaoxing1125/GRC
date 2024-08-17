import numpy as np
import pickle
import copy

class Driver():
    def __init__(self, did, gid, grid):
        self.did = did
        self.gid = gid
        self.grid = grid
        self.grid.driver_list.append(self.did)

    def update(self, Grids):
        self.gid = self.order.end_grid
        self.order = None

        self.grid = Grids[self.gid]
        self.grid.driver_list.append(self.did)


class Order():
    def __init__(self, order_idx, start_time, end_time, start_grid, end_grid, price):
        self.order_idx = order_idx
        self.start_time = start_time 
        self.end_time = end_time 
        self.start_grid = start_grid
        self.end_grid = end_grid
        self.price = price

class Grid():
    def __init__(self, idx, gps_list):
        self.idx = idx

        self.gps_list = copy.deepcopy(gps_list)

        lons = [x for id, x in enumerate(gps_list) if id % 2 == 0]
        lats = [x for id, x in enumerate(gps_list) if id % 2 == 1]

        self.center = [np.mean(lons), np.mean(lats)]

        self.order_dict = None

        self.driver_list = []
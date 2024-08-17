import numpy as np
import pickle
import copy
import random
from objects import *
from utils import *

np.random.seed(100)
random.seed(100)

class ENV():
    def __init__(self, Orders_Dataset, Grids, Drivers_Dataset):
        self.Orders_Dataset = Orders_Dataset
        self.Grids = Grids
        self.Drivers_Dataset = Drivers_Dataset

        self.active_drivers = copy.deepcopy(Drivers_Dataset)
        self.inactive_drivers = {}

        self.active_orders = {}

        self.matched_orders = []

    def step(self, time):

        canceled_num = old_orders_cancel(self.active_orders, time)

        old_orders_finish(self.inactive_drivers, self.active_drivers, time, Grids)
        
        new_orders_num = load_new_orders(self, time)

        
        revenue, matched_orders_num = match(self.active_orders, self.active_drivers, self.inactive_drivers, self.Grids, self.matched_orders)

        reposition(self.active_drivers)

        print("Info:", time, int(time / 3600), int(time / 60) - int(time / 3600) * 60, revenue, matched_orders_num, len(self.active_drivers), len(self.inactive_drivers), len(self.active_orders), canceled_num, new_orders_num)

        return revenue, new_orders_num, matched_orders_num


if __name__ == "__main__":
    for date_idx in range(1, 4, 1):
        Grids_Dataset  = pickle.load(open('../dataset/Grids_Dataset', 'rb'))
        Grids = {}
        for key, value in Grids_Dataset.items():
            Grids[key] = Grid(value[0], value[1])

        Orders_Dataset = pickle.load(open('../dataset/Orders_Dataset_2016110'+str(date_idx)+'_i2s', 'rb'))


        start_hour = 7
        Drivers_Dataset = Sample_Driver(2000, len(Grids), Grids, '2016110'+str(date_idx), start_hour)

        env = ENV(Orders_Dataset, Grids, Drivers_Dataset)
        revenue, new_orders_num, matched_orders_num = 0, 0, 0
        for time in range(start_hour * 60 * 60, 10 * 60 * 60, 2):
            _revenue, _new_orders_num, _matched_orders_num = env.step(time)
            revenue += _revenue
            new_orders_num += _new_orders_num
            matched_orders_num += _matched_orders_num

        print('2016110'+str(date_idx), " Revenue & ORR:", revenue,  matched_orders_num / new_orders_num)
        pickle.dump(env.matched_orders, open('./result/matched_orders_2016110'+str(date_idx), 'wb+'))



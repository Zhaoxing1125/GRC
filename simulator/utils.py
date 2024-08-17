import numpy as np
import random
import copy
import pickle
from objects import *
from geopy import distance
from coordinate import *
from solver import *
import torch

def old_orders_cancel(active_orders, time):
    cancel_list = []
    for oid, order in active_orders.items():
        if time - order.start_time >= 60 * 3:
            cancel_list.append(oid)

    for oid in cancel_list:
        active_orders.pop(oid)

    return len(cancel_list)

def get_grid_state(end_grid, active_orders, active_drivers):
    state = []

    d_num = 0
    for driver in active_drivers.values():
        if driver.gid == end_grid:
            d_num += 1
    
    o_list = []
    for order in active_orders.values():
        if order.start_grid == end_grid:
            o_list.append([order.price, order.price / (order.end_time - order.start_time)])

    if len(o_list) == 0:
        return [d_num, 0.0, 0.0, 0.0, 0.0]
    
    pv = [x[0] for x in o_list]
    pvp = [x[1] for x in o_list]

    return [d_num-len(o_list), np.mean(pv), np.min(pv), np.max(pv), np.mean(pvp)]

def match(active_orders, active_drivers, inactive_drivers, Grids, matched_orders, MDP_TABLE=None):

    d_can_match, o_can_match, edge_weight, price_dict = [], [], {}, {}
    for oid, order in active_orders.items():
        for did, driver in active_drivers.items():
            o_grid_gps = Grids[order.start_grid].center
            d_grid_gps = driver.grid.center

            src_x, src_y = gcj02_to_wgs84(o_grid_gps[0], o_grid_gps[1])
            dst_x, dst_y = gcj02_to_wgs84(d_grid_gps[0], d_grid_gps[1])
            dist_osrm = distance.distance([src_y, src_x], [dst_y, dst_x]).km

            if dist_osrm < 3.0:
                o_can_match.append(oid)
                d_can_match.append(did)

                edge_weight[(oid, did)] = order.price 
                price_dict[(oid, did)] = order.price 
                if MDP_TABLE is not None:
                    if type(MDP_TABLE) is not dict:
                        state = get_grid_state(order.end_grid, active_orders, active_drivers)
                        time_step = [0.0] * 18
                        if int(order.end_time/600) - 7*6 >= 18:
                            pass
                        else:
                            time_step[int(order.end_time/600) - 7*6] = 1.0
                            state.extend(time_step)
                            act = MDP_TABLE(torch.tensor(state).float())
                            edge_weight[(oid, did)] += act.item() * 0.99 ** (int(order.end_time/600) - int(order.start_time/600))
                    else:
                        key = (order.end_grid, int(order.end_time/600))
                        if key in MDP_TABLE:
                            edge_weight[(oid, did)] += MDP_TABLE[key] * 0.99 ** (int(order.end_time/600) - int(order.start_time/600))
    
    _, _, pair_map, revenue = km_match(d_can_match, o_can_match, edge_weight, price_dict)

    for did, oid in pair_map.items():
        driver = active_drivers[did]
        order = active_orders[oid]

        driver.order = copy.deepcopy(order)

        inactive_drivers[did] = driver
        active_drivers.pop(did)
        active_orders.pop(oid)

        matched_orders.append(
            [
                order.start_grid,
                order.start_time,
                order.end_grid,
                order.end_time,
                order.price
            ]
        )

    return revenue, len(pair_map)

def count_drivers_num(active_drivers, stats_active_drivers, time):
    time_step = int(time / 60 / 10)
    if time_step not in stats_active_drivers:
        stats_active_drivers[time_step] = {}
    for did, driver in active_drivers.items():
        gid = driver.gid
        if gid not in stats_active_drivers[time_step]:
            stats_active_drivers[time_step][gid] = []
        if did not in stats_active_drivers[time_step][gid]:
            stats_active_drivers[time_step][gid].append(did)

def count_orders_num(active_orders, stats_active_orders, time):
    time_step = int(time / 60 / 10)
    if time_step not in stats_active_orders:
        stats_active_orders[time_step] = {}
    for oid, order in active_orders.items():
        gid = order.start_grid
        if gid not in stats_active_orders[time_step]:
            stats_active_orders[time_step][gid] = {}
        if oid not in stats_active_orders[time_step][gid]:
            stats_active_orders[time_step][gid].update({
                oid: [
                order.start_grid,
                order.start_time,
                order.end_grid,
                order.end_time,
                order.price
            ]}
            )

def load_new_orders(env, time):
    if time not in env.Orders_Dataset:
        return 0
    
    Orders_Dataset = env.Orders_Dataset[time]
    
    curr_orders_idx, new_orders_num = len(env.active_orders), 0
    for gid, order_list in Orders_Dataset.items():
        if np.random.rand() > 0.5:
            continue
        for order in order_list:
            order_idx, start_time, end_time, start_grid, end_grid, price = order
            env.active_orders[curr_orders_idx] = Order(order_idx, start_time, end_time, start_grid, end_grid, price)
            curr_orders_idx += 1

            new_orders_num += 1

    return new_orders_num

def old_orders_finish(inactive_drivers, active_drivers, time, Grids):
    invalid_drivers = []
    for did, driver in inactive_drivers.items():
        if driver.order.end_time == time:
            invalid_drivers.append(did)
            active_drivers[did] = driver
            driver.update(Grids)

    for did in invalid_drivers:
        inactive_drivers.pop(did)

def reposition(active_drivers):
    pass
    #return active_drivers

def Sample_Driver(number, all_grids_num, Grids, date_t, start_hour):
    Orders_Dataset = pickle.load(open('../dataset/Orders_Dataset_'+date_t, 'rb'))
    orders = Orders_Dataset[ start_hour * 6]

    prob = []
    for gid in range(all_grids_num):
        if gid in orders:
            prob.append(len(orders[gid]))
        else:
            prob.append(0.0)

    if sum(prob) > 0:
        prob = np.array(prob)
        prob = prob/prob.sum()
        sample_list = np.random.choice(all_grids_num, number, p=prob)
    else:
        sample_list = np.random.choice(all_grids_num, number)

    drivers = {}
    for did, gid in enumerate(sample_list):
        drivers[did] = Driver(did, gid, Grids[gid])

    return drivers

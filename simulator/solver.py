from pyscipopt import Model
#import gurobipy as gp
#from gurobipy import GRB

def km_match_gp(driver_id_list, order_id_list, order_price_dict, price_dict):
    sol_dict = {}
    
    m = gp.Model()
    # varibles
    m.modelSense, obj = GRB.MAXIMIZE, 0

    var_dict, cc, all_prices, d_dict, o_dict = {}, 0, {}, {}, {}
    for i in range(len(driver_id_list)):
        
        d = driver_id_list[i]
        o = order_id_list[i]
        p = order_price_dict[(o, d)]

        var_dict[cc] = m.addVars(1, lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="t_x_"+str(d)+"_"+str(o))#model.addVar("x_"+str(d)+"_"+str(o), vtype="INTEGER")
        
        #all_prices[cc] = p
        obj += var_dict[cc][0] * p
        #m.addConstr(var_dict[cc][0] >= 0)
        #m.addConstr(var_dict[cc][0] <= 1)
        if d not in d_dict:
            d_dict[d] = [var_dict[cc][0]]
        else:
            d_dict[d].append(var_dict[cc][0])
        if o not in o_dict:
            o_dict[o] = [var_dict[cc][0]]
        else:
            o_dict[o].append(var_dict[cc][0])
        cc += 1

    for d in d_dict:
        m.addConstr(sum([x for x in d_dict[d]]) >= 0)
        m.addConstr(sum([x for x in d_dict[d]]) <= 1)
    for o in o_dict:
        m.addConstr(sum([x for x in o_dict[o]]) >= 0)
        m.addConstr(sum([x for x in o_dict[o]]) <= 1)
    
    m.setObjective(obj)
    m.Params.LogToConsole = 0
    m.optimize()

    sol_dict = {}
    for v in m.getVars():
        name = v.VarName
        sol_dict[name[:-3]] = v.X

    
    assigned_d_ids, assigned_o_ids = [], []
    pair_map = {}
    rew = 0.0
    for name, value in sol_dict.items():
        a = name.split('_')
        d_id, o_id = int(a[2]), int(a[3])
        if value == 1:
            assigned_d_ids.append(d_id)
            assigned_o_ids.append(o_id)
            pair_map[d_id] = o_id
            rew += price_dict[(o_id, d_id)]

    return assigned_d_ids, assigned_o_ids, pair_map, rew

def km_match(driver_id_list, order_id_list, order_price_dict, price_dict):
    sol_dict = {}
    
    model = Model('slove')
    var_dict, cc, all_prices, d_dict, o_dict, var_name = {}, 0, {}, {}, {}, {}
    for i in range(len(driver_id_list)):
        
        d = driver_id_list[i]
        o = order_id_list[i]
        p = order_price_dict[(o, d)]

        var_dict[cc] = model.addVar("x_"+str(d)+"_"+str(o), vtype="INTEGER")
        var_name["t_x_"+str(d)+"_"+str(o)] = var_dict[cc]
        
        all_prices[cc] = p
        model.addCons(var_dict[cc] >= 0)
        model.addCons(var_dict[cc] <= 1)
        if d not in d_dict:
            d_dict[d] = [var_dict[cc]]
        else:
            d_dict[d].append(var_dict[cc])
        if o not in o_dict:
            o_dict[o] = [var_dict[cc]]
        else:
            o_dict[o].append(var_dict[cc])
        cc += 1

    for d in d_dict:
        model.addCons(sum([x for x in d_dict[d]]) >= 0)
        model.addCons(sum([x for x in d_dict[d]]) <= 1)
    for o in o_dict:
        model.addCons(sum([x for x in o_dict[o]]) >= 0)
        model.addCons(sum([x for x in o_dict[o]]) <= 1)
    
    model.setObjective(sum([a*all_prices[a_id] for a_id, a in var_dict.items()]), 'maximize')
    model.hideOutput(True)
    model.optimize()

    sol_dict = {}
    for name, var in var_name.items():
        sol_dict[name] = model.getVal(var)
    
    assigned_d_ids, assigned_o_ids = [], []
    pair_map = {}
    rew = 0.0
    for name, value in sol_dict.items():
        a = name.split('_')
        d_id, o_id = int(a[2]), int(a[3])
        if value == 1:
            assigned_d_ids.append(d_id)
            assigned_o_ids.append(o_id)
            pair_map[d_id] = o_id
            rew += price_dict[(o_id, d_id)]

    return assigned_d_ids, assigned_o_ids, pair_map, rew
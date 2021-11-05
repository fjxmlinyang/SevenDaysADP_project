import pandas as pd
from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import numpy as np
def Perfect_Opt(Input_folder, scenario):
    print('################################## LAC_PSH_profit_max ###################################')
    ################################### Read Data ####################################

    ##read PSH
    filename = Input_folder + '/PSH.csv'
    Data = pd.read_csv(filename)
    df=pd.DataFrame(Data)
    PSHmin_g = float(df['GenMin'])
    PSHmax_g = float(df['GenMax'])
    PSHmin_p = float(df['PumpMin'])
    PSHmax_p = float(df['PumpMax'])
    PSHefficiency = list(df['Efficiency'])
    PSHname = list(df['Name'])

    ##read E
    filename = Input_folder + '/Reservoir.csv'
    Data = pd.read_csv(filename)
    df=pd.DataFrame(Data)
    Emin = float(df['Min'])
    Emax = float(df['Max'])
    Edayend= float(df['End'])
    Estart = float(df['Start'])
    Ename = df['Name']


    ##read price
    filename = Input_folder + '/DECO_prd_dataframe_Perfect_October 1521 2019.csv'
    Data = pd.read_csv(filename)
    df = pd.DataFrame(Data)
    Column_name = list(Data.columns)
    Column_name = Column_name[scenario]
    lmp_quantiles = [1]
    lmp_scenarios = []
    Nlmp_s = 1

    # the scenario you want
    lmp_scenarios.append(df[Column_name])
    lmp_scenarios = list(lmp_scenarios[0])


    e_time_periods = [] # 这个比总时间小1
    for i in range(1, len(lmp_scenarios)):
    #for i in range(len(lmp_scenarios)):
        e_time_periods.append('T' + str(i))


    ################################### Build Model ####################################
    model = Model('DAMarket')

    # psh0 and e0 are the first stage deterministic variables, while psh and e are stochastic variables.
    psh0_gen= model.addVars(list(PSHname), ub=float('inf'), lb=-float('inf'), name='PSH0_gen')
    psh0_pump= model.addVars(list(PSHname), ub=float('inf'), lb=-float('inf'), name='PSH0_pump')

    psh_gen = model.addVars(range(Nlmp_s), e_time_periods, list(PSHname),  ub = float('inf'), lb = -float('inf'), name = 'PSH_gen')
    psh_pump = model.addVars(range(Nlmp_s), e_time_periods, list(PSHname),  ub = float('inf'), lb = -float('inf'), name = 'PSH_pump')

    e0 = model.addVars(list(Ename), ub = float('inf'), lb = -float('inf'), name='E0')
    e = model.addVars(range(Nlmp_s), e_time_periods, list(Ename),  ub = float('inf'), lb = -float('inf'), name='E')

    model.update()

    ### add constraints

    # fix start state of energy stored in reservoir
    for j in Ename:
        print('Estart:', float(Estart))
        LHS = e0[j] + grb.quicksum(psh0_gen[j]/PSHefficiency[PSHname.index(j)] for j in PSHname)\
              - grb.quicksum(psh0_pump[j]*PSHefficiency[PSHname.index(j)] for j in PSHname) - Estart
        RHS = 0
        print(LHS, RHS)
        model.addConstr(LHS == RHS, name='%s_%s' % ('SOC0', j))


    # state of charge
    for i in e_time_periods:
        time_id = e_time_periods.index(i)
        if time_id == 0: #相当于除了e0的第一个，需要和e0交换
            for s in range(Nlmp_s):
                for j in Ename:
                    LHS = e[s, i, j] + grb.quicksum(psh_gen[s, i, j]/PSHefficiency[PSHname.index(j)] for j in PSHname)\
                    - grb.quicksum(psh_pump[s, i, j] * PSHefficiency[PSHname.index(j)] for j in PSHname)- e0[j]
                    RHS = 0
                    model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, time_id, s))
        else:
            for s in range(Nlmp_s):
                for j in Ename:
                    time_previous = e_time_periods[time_id - 1]
                    LHS=e[s,i,j] + grb.quicksum(psh_gen[s, i, j]/PSHefficiency[PSHname.index(j)] for j in PSHname)\
                    - grb.quicksum(psh_pump[s, i, j]*PSHefficiency[PSHname.index(j)] for j in PSHname)-e[s, time_previous, j]
                    RHS= 0
                    model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, time_id, s))
            if time_id == len(e_time_periods)-1: #这个必须得在这里，因为这是最后一个多加了一个
                for s in range(Nlmp_s):
                    for j in Ename:
                        LHS = Edayend - e[s, i, j]
                        RHS = 0
                        model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, len(e_time_periods), s))


    # Upper and lower bounds
    for j in PSHname:
        model.addConstr(psh0_gen[j] <= PSHmax_g, name='%s_%s' % ('psh_gen_max0', j))
        model.addConstr(psh0_gen[j] >= PSHmin_g, name='%s_%s' % ('psh_gen_min0', j))
        model.addConstr(psh0_pump[j] <= PSHmax_p, name='%s_%s' % ('psh_pump_max0', j))
        model.addConstr(psh0_pump[j] >= PSHmin_p, name='%s_%s' % ('psh_pump_min0', j))
    for k in Ename:
        model.addConstr(e0[k] <= Emax, name='%s_%s' % ('e_max0', k))
        model.addConstr(e0[k] >= Emin, name='%s_%s' % ('e_min0', k))
    for s in range(Nlmp_s):
        for i in e_time_periods:
            for j in PSHname:
                model.addConstr(psh_gen[s, i, j] <= PSHmax_g, name='%s_%s_%s_%s' % ('psh_gen_max',s, i, j))
                model.addConstr(psh_gen[s, i, j] >= PSHmin_g, name='%s_%s_%s_%s' % ('psh_gen_min',s, i, j))
                model.addConstr(psh_pump[s, i, j] <= PSHmax_p, name='%s_%s_%s_%s' % ('psh_pump_max', s, i, j))
                model.addConstr(psh_pump[s, i, j] >= PSHmin_p, name='%s_%s_%s_%s' % ('psh_pump_min', s, i, j))
            for k in Ename:
                model.addConstr(e[s, i, k] <= Emax, name='%s_%s_%s_%s' % ('e_max',s, i, k))
                model.addConstr(e[s, i, k] >= Emin, name='%s_%s_%s_%s' % ('e_min',s, i, k))

    ### Objective function
    psh_max=[]
    for s in range(Nlmp_s):
        p_s = lmp_quantiles[s]
        for j in PSHname:
            psh_max.append((psh0_gen[j] - psh0_pump[j]) * lmp_scenarios[0] * p_s)
            for i in e_time_periods:
                _temp = e_time_periods.index(i)+1
                lac_lmp = lmp_scenarios[_temp]   #why这里加上1
                psh_max.append((psh_gen[s, i, j] - psh_pump[s, i, j]) * lac_lmp * p_s)
    obj = quicksum(psh_max)

    ################################### Solve Model ####################################

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    ####print the first result
    obj = model.getObjective()  # self.calculate_pts(self.optimal_soc_sum)
    # self.optimal_profit = obj.getValue()
    print('the optimal profit is #####', obj.getValue())
    print('#####')


    # print results for variables
    for v in model.getVars():
        print("%s %f" % (v.Varname, v.X))


    ## add psh
    psh0=[]
    PSH0=[]
    for v in [v for v in model.getVars() if 'PSH0_gen' in v.Varname]:
        psh = v.X
        psh0.append(psh)
    for v in [v for v in model.getVars() if 'PSH0_pump' in v.Varname]:
        psh = v.X
        psh0.append(-psh)
    PSH0.append(sum(psh0))
    #if LAC_last_windows:
    psh_last_window_gen=[]
    psh_last_window_pump=[]
    PSH=[PSH0[0]]

    ## add soc
    SOC=[]

    for v in [v for v in model.getVars() if 'PSH_gen' in v.Varname]:
        psh = v.X
        psh_last_window_gen.append(psh)

    for v in [v for v in model.getVars() if 'PSH_pump' in v.Varname]:
        psh = v.X
        psh_last_window_pump.append(psh)

    for i in range(len(psh_last_window_gen)):
        PSH.append(psh_last_window_gen[i]-psh_last_window_pump[i])

    for v in [v for v in model.getVars() if 'E' in v.Varname]:
        soc = v.X
        SOC.append(soc)


    return SOC, PSH, lmp_scenarios


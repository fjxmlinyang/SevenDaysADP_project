import pandas as pd
from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import numpy as np
def Perfect_Opt(LAC_bhour,LAC_last_windows,Input_folder,Output_folder,date,RT_DA,probabilistic, time_total, scenario):
    print('################################## LAC_PSH_profit_max #', LAC_bhour, 'probabilistic=',probabilistic,' ##################################')
    ################################### Read Data ####################################
    filename = Input_folder + '/PSH.csv'
    Data = pd.read_csv(filename)
    df=pd.DataFrame(Data)
    PSHmin_g = df['GenMin']
    PSHmax_g = df['GenMax']
    PSHmin_p = df['PumpMin']
    PSHmax_p = df['PumpMax']
    PSHcost = df['Cost']
    PSHefficiency = list(df['Efficiency'])
    PSHname = list(df['Name'])
    filename = Input_folder + '/Reservoir.csv'
    Data = pd.read_csv(filename)
    df=pd.DataFrame(Data)
    Emin = df['Min']
    Emax = df['Max']
    Ename = df['Name']
    Edayend= float(df['End'])
    if LAC_bhour==0 or LAC_last_windows:
        Estart = df['Start']
    else:
        filename = Output_folder+'/LAC_Solution_System_SOC_' + str(LAC_bhour-1) + '.csv'
        Data = pd.read_csv(filename)
        df = pd.DataFrame(Data)
        Estart = df['SOC'][0]
        print('here is ####', Estart)

    if LAC_last_windows:

        filename = Input_folder + '/Full_opt_wlen_' + str(time_total+1) + '_' + date + '_50' + '.csv'



    Data = pd.read_csv(filename)
    df = pd.DataFrame(Data)
    Column_name=list(Data.columns)
    lmp_quantiles = []
    lmp_scenarios = []
    DA_lmp=[]
    Nlmp_s = 1
    # probability of each scenario is evenly distributed
    lmp_quantiles.append(1.0 / Nlmp_s)
    _temp_name = 'V' + str(scenario)
    lmp_scenarios.append(list(df[_temp_name]))


    e_time_periods = []
    for i in range(1,len(lmp_scenarios[0])):
        e_time_periods.append('T' + str(i))
    ################################### Build Model ####################################
    model = Model('DAMarket')
    ### define variables
    psh_max_g={}
    psh_min_g={}
    psh_max_p = {}
    psh_min_p = {}
    e_max={}
    e_min={}

    for i in e_time_periods:
        for k in PSHname:
            for s in range(Nlmp_s):
                psh_max_g[(s,i,k)]=list(PSHmax_g)[list(PSHname).index(k)]
                psh_min_g[(s,i,k)]=list(PSHmin_g)[list(PSHname).index(k)]
                psh_max_p[(s, i, k)] = list(PSHmax_p)[list(PSHname).index(k)]
                psh_min_p[(s, i, k)] = list(PSHmin_p)[list(PSHname).index(k)]
    for i in e_time_periods:
        for f in Ename:
            for s in range(Nlmp_s):
                e_max[(s,i,f)]=list(Emax)[list(Ename).index(f)]
                e_min[(s,i,f)]=list(Emin)[list(Ename).index(f)]

    e_max_inf = {}
    e_min_inf = {}
    psh_max_inf = {}
    psh_min_inf = {}

    for i in e_time_periods:
        for k in PSHname:
            for s in range(Nlmp_s):
                psh_max_inf[(s, i, k)] = float('inf')
                psh_min_inf[(s, i, k)] = -float('inf')

    for i in e_time_periods:
        for f in Ename:
            for s in range(Nlmp_s):
                e_max_inf[(s, i, f)] = float('inf')
                e_min_inf[(s, i, f)] = -float('inf')

    # psh0 and e0 are the first stage deterministic variables, while psh and e are stochastic variables.
    psh0_gen= model.addVars(list(PSHname),ub=float('inf'),lb=-float('inf'),name='PSH0_gen')
    psh0_pump= model.addVars(list(PSHname),ub=float('inf'),lb=-float('inf'),name='PSH0_pump')

    psh_gen = model.addVars(range(Nlmp_s),e_time_periods,list(PSHname),ub=psh_max_inf,lb=psh_min_inf,name='PSH_gen')
    psh_pump = model.addVars(range(Nlmp_s),e_time_periods,list(PSHname),ub=psh_max_inf,lb=psh_min_inf,name='PSH_pump')

    e0 = model.addVars(list(Ename),ub=float('inf'),lb=-float('inf'),name='E0')
    e = model.addVars(range(Nlmp_s),e_time_periods,list(Ename),ub=e_max_inf,lb=e_min_inf,name='E')

    model.update()

    ### add constraints

    # fix start state of energy stored in reservoir
    for j in Ename:
        print('Estart:', float(Estart))
        LHS = e0[j] + grb.quicksum(psh0_gen[j]/PSHefficiency[PSHname.index(j)] for j in PSHname)\
              - grb.quicksum(psh0_pump[j]*PSHefficiency[PSHname.index(j)] for j in PSHname) - float(Estart)
        RHS = 0
        print(LHS, RHS)
        model.addConstr(LHS == RHS, name='%s_%s' % ('SOC0', j))
    # state of charge
    for i in e_time_periods:
        time_id=e_time_periods.index(i)
        if time_id == 0:
            for s in range(Nlmp_s):
                for j in Ename:
                    print('Estart:',float(Estart))
                    LHS = e[s,i, j] + grb.quicksum(psh_gen[s,i,j]/PSHefficiency[PSHname.index(j)] for j in PSHname)\
                    - grb.quicksum(psh_pump[s,i,j]*PSHefficiency[PSHname.index(j)] for j in PSHname)-e0[j]
                    RHS = 0
                    print(LHS,RHS)
                    model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, time_id, s))
                if time_id == len(e_time_periods) - 1:
                    for j in Ename:
                        print('Edayend:', Edayend)
                        LHS = Edayend - e[s, i, j]
                        RHS = 0
                        print(LHS,RHS)
                        model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, len(e_time_periods),s))
        else:
            for s in range(Nlmp_s):
                time_previous=e_time_periods[time_id-1]
                for j in Ename:
                    LHS=e[s,i,j] + grb.quicksum(psh_gen[s,i,j]/PSHefficiency[PSHname.index(j)] for j in PSHname)\
                    - grb.quicksum(psh_pump[s,i,j]*PSHefficiency[PSHname.index(j)] for j in PSHname)-e[s,time_previous,j]
                    RHS= 0
                    print(LHS, RHS)
                    model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, time_id, s))
                if time_id == len(e_time_periods) - 1:
                    for j in Ename:
                        print('Edayend:', Edayend)
                        LHS = Edayend - e[s, i, j]
                        RHS = 0
                        print(LHS,RHS)
                        model.addConstr(LHS == RHS, name='%s_%s_%d_%d' % ('SOC', j, len(e_time_periods),s))

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
                model.addConstr(psh_gen[s, i, j] <= psh_max_g[s, i, j], name='%s_%s_%s_%s' % ('psh_gen_max',s, i, j))
                model.addConstr(psh_gen[s, i, j] >= psh_min_g[s, i, j], name='%s_%s_%s_%s' % ('psh_gen_min',s, i, j))
                model.addConstr(psh_pump[s, i, j] <= psh_max_p[s, i, j], name='%s_%s_%s_%s' % ('psh_pump_max', s, i, j))
                model.addConstr(psh_pump[s, i, j] >= psh_min_p[s, i, j], name='%s_%s_%s_%s' % ('psh_pump_min', s, i, j))
            for k in Ename:
                model.addConstr(e[s, i, k] <= e_max[s, i, k], name='%s_%s_%s_%s' % ('e_max',s, i, k))
                model.addConstr(e[s, i, k] >= e_min[s, i, k], name='%s_%s_%s_%s' % ('e_min',s, i, k))

    ### Objective function
    psh_max=[]
    for s in range(Nlmp_s):
        p_s = lmp_quantiles[s]
        for j in PSHname:
            psh_max.append((psh0_gen[j] - psh0_pump[j]) * lmp_scenarios[s][0] * p_s)
            for i in e_time_periods:
                    lac_lmp = lmp_scenarios[s][e_time_periods.index(i)+1]
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

    ################################### return solutions ####################################

    SOC0=[]
    filename=Output_folder+'/LAC_Solution_System_SOC_'+str(LAC_bhour)+'.csv'
    with open(filename, 'w') as wf:
        wf.write('Num_Period,Reservoir_Name,SOC\n')
        for v in [v for v in model.getVars() if ('Reservoir' in v.Varname and 'E0' in v.Varname)]:
            if 'Reservoir' in v.Varname:
                soc=v.X
                SOC0.append(soc)
                time='T'+str(LAC_bhour)
                name=v.Varname[3:13]
                st = time + ',' + '%s,%.1f' % (name,soc) + '\n'
                wf.write(st)
    psh0=[]
    PSH0=[]
    for v in [v for v in model.getVars() if 'PSH0_gen' in v.Varname]:
        psh = v.X
        psh0.append(psh)
    for v in [v for v in model.getVars() if 'PSH0_pump' in v.Varname]:
        psh = v.X
        psh0.append(-psh)
    PSH0.append(sum(psh0))
    if LAC_last_windows:
        psh_last_window_gen=[]
        psh_last_window_pump=[]
        PSH=[PSH0[0]]
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
    else:
        return SOC0, PSH0, lmp_scenarios

# def PSH_profitmax_plot(Input_folder,Output_folder,date,RT_DA,probabilistic):
#     filename=Output_folder+'/PSH_Profitmax_Rolling_Results_'+date+'.csv'
#     Data = pd.read_csv(filename)
#     df = pd.DataFrame(Data)
#     SOC_after_fact=df['After_fact_SOC']
#     SOC_rolling=df['SOC_Results']
#     DA_SOC=df['DA_SOC']
#     PSH_after_fact=df['After_fact_PSH']
#     PSH_rolling=df['PSH_Results']
#     DA_PSH=df['DA_PSH']
#
#     filename = Input_folder + '/prd_dataframe_wlen_24_'+date+'.csv'
#     Data = pd.read_csv(filename)
#     df = pd.DataFrame(Data)
#
#     lmp_hindsight=list(df['RT_LMP'])
#     DA_LMP=list(df['DA_LMP'])
#     Afterfact_profit=0
#     Rolling_profit=0
#     DA_profit=0
#     for i in range(len(PSH_rolling)):
#         Afterfact_profit+=PSH_after_fact[i] * lmp_hindsight[i]
#         DA_profit+=DA_PSH[i]*lmp_hindsight[i]
#         Rolling_profit += PSH_rolling[i] * lmp_hindsight[i]
#     if probabilistic:
#         filename = Output_folder + '/PSH_Profitmax_Profits_'+date+'.csv'
#     else:
#         filename = Output_folder + '/PSH_Profitmax_Profits_Single_prd_' + date + '.csv'
#     with open(filename, 'w') as wf:
#         wf.write('After The Fact Profits, Rolling Window Profits, Stay with DA Profits\n')
#         st ='%.1f,%.1f,%.1f' % (Afterfact_profit, Rolling_profit, DA_profit) + '\n'
#         wf.write(st)
#
#     fig, ax1 = plt.subplots()
#     afterfact_color='darkorange'
#     da_color='tomato'
#     rolling_color='royalblue'
#     if RT_DA:
#         soc_afterfact_plot=ax1.plot(SOC_after_fact[:len(SOC_rolling)],color=afterfact_color,alpha=0)
#     else:
#         da_soc_plot=ax1.plot(DA_SOC[:len(SOC_rolling)],color=da_color,alpha=0)
#     soc_rolling_plot = ax1.plot(SOC_rolling,color=rolling_color,alpha=0)
#     x=np.arange(0,len(SOC_rolling),1)
#     if RT_DA:
#         soc_afterfact_area=ax1.fill_between(x,0,SOC_after_fact[:len(SOC_rolling)],alpha=0.3,color=afterfact_color)
#     else:
#         da_soc_area=ax1.fill_between(x,0,DA_SOC[:len(SOC_rolling)],alpha=0.3,color=da_color)
#     soc_rolling_area=plt.fill_between(x,0,SOC_rolling,alpha=0.3,color=rolling_color)
#     ax1.set_xlabel('Hour')
#     ax1.set_ylabel('SOC [MWh]')
#     plt.yticks(np.arange(1000, 4000, 500))
#
#     ax2= ax1.twinx()
#     Data={'LMP':lmp_hindsight[:len(SOC_rolling)],'DA_LMP':DA_LMP[:len(SOC_rolling)],}
#     df=pd.DataFrame(Data)
#     lmp_plot=df['LMP']
#     afterfact_lmp=ax2.plot(lmp_plot*10+600,color='limegreen',linestyle='--')
#     if RT_DA==0:
#         DA_lmp_plot=df['DA_LMP']
#         DA_lmp = ax2.plot(DA_lmp_plot * 10 + 600, color='olivedrab', linestyle='--')
#
#     psh_linestyle='-'
#     if RT_DA:
#         psh_afterfact_plot = ax2.plot(PSH_after_fact[:len(PSH_rolling)],color=afterfact_color,linestyle=psh_linestyle)
#     else:
#         da_psh_plot = ax2.plot(DA_PSH[:len(PSH_rolling)],color=da_color,linestyle=psh_linestyle)
#
#     psh_rolling_plot = ax2.plot(PSH_rolling,color=rolling_color,linestyle=psh_linestyle)
#     psh_idle_plot = ax2.plot([0]*len(PSH_rolling),color='black',linestyle='-.')
#
#
#
#     ax2.set_ylabel('PSH Output [MW]')
#     plt.yticks(np.arange(-300, 1800, 500))
#     if RT_DA:
#         plt.legend((soc_afterfact_area, soc_rolling_area,psh_afterfact_plot[0], psh_rolling_plot[0],afterfact_lmp[0]),
#                ('SOC after fact', 'SOC rolling','PSH after fact', 'PSH rolling','Actual RT LMP'),loc='upper right')
#     else:
#         plt.legend((da_soc_area, soc_rolling_area, da_psh_plot[0], psh_rolling_plot[0], afterfact_lmp[0], DA_lmp[0]),
#                    ('SOC DA', 'SOC rolling', 'PSH DA', 'PSH rolling', 'Actual RT LMP', 'DA LMP'),
#                    loc='upper right')
#     if RT_DA:
#         if probabilistic:
#             filename=Output_folder+'/PSH profitmax SOC results benchmark to RT after the fact_'+date
#         else:
#             filename = Output_folder + '/PSH profitmax SOC single prd results benchmark to RT after the fact_' + date
#     else:
#         if probabilistic:
#             filename=Output_folder+'/PSH profitmax SOC results benchmark to DA_'+date
#         else:
#             filename = Output_folder + '/PSH profitmax SOC single prd results benchmark to DA_' + date
#
#     plt.savefig(filename,dpi=300)
#     plt.show()
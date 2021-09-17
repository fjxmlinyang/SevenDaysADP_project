import pandas as pd
from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import numpy as np
from SystemSetUp import *
from CurrModelPara import *
from Curve import *

class OptModelSetUp():
    def __init__(self):
        self.gur_model = None
        self.psh_system = None
        self.e_system = None
        self.lmp = None
        self.curve = None
        self.curr_model_para = None
        # self.pre_curve = pre_curve
        # self.pre_lmp = pre_lmp

    # def __init__(self, psh_system, e_system, lmp, curve, curr_model_para, gur_model):
    #     self.gur_model = gur_model
    #     self.psh_system = psh_system
    #     self.e_system = e_system
    #     self.lmp = lmp
    #     self.curve = curve
    #     self.curr_model_para = curr_model_para
    #     # self.pre_curve = pre_curve
    #     # self.pre_lmp = pre_lmp

########################################
########################################
#funtions for set up
    def add_var_e(self, var_name):
        return self.gur_model.addVars(self.e_system.parameter['EName'], ub=float('inf'),lb=-float('inf'), vtype="C", name=var_name)

    def add_var_I(self, var_name):
        return self.gur_model.addVars(self.e_system.parameter['EName'], vtype="B", name=var_name)

    def add_var_psh(self, var_name):
        return self.gur_model.addVars(self.psh_system.parameter['PSHName'], ub=float('inf'),lb=-float('inf'),vtype="C",name=var_name)



    def add_constraint_rolling(self):
        ## SOC0: e_0=E_start; loop from 0 to 22; e_1=e_0+psh1;....e_23=e_22+psh_23; when loop to 22; directly add e_23=E_end
        for k in self.e_system.parameter['EName']:
            print('Estart:', float(self.e_system.parameter['EStart']))
            LHS = self.e[k] + grb.quicksum(self.psh_gen[j] / self.psh_system.parameter['GenEfficiency'] for j in self.psh_system.parameter['PSHName']) \
                                          - grb.quicksum(self.psh_pump[j] * self.psh_system.parameter['PumpEfficiency'] for j in self.psh_system.parameter['PSHName'])
            RHS = self.e_system.parameter['EStart']
            print(LHS)
            ###if we calculate the first one, we use 'SOC0', and the last we use 'End'; or we choose the SOC0 to "beginning", at the same time the last we use 'SOC'.
            self.gur_model.addConstr(LHS == RHS, name='%s_%s' % ('SOC0', k))


    def add_constraint_epsh(self):
        for j in self.psh_system.parameter['PSHName']:  # all are lists
            self.gur_model.addConstr(self.psh_gen[j] <= self.psh_system.parameter['GenMax'], name='%s_%s' % ('psh_gen_max0', j))
            self.gur_model.addConstr(self.psh_gen[j] >= self.psh_system.parameter['GenMin'], name='%s_%s' % ('psh_gen_min0', j))
            self.gur_model.addConstr(self.psh_pump[j] <= self.psh_system.parameter['PumpMax'], name='%s_%s' % ('psh_pump_max0', j))
            self.gur_model.addConstr(self.psh_pump[j] >= self.psh_system.parameter['PumpMin'], name='%s_%s' % ('psh_pump_min0', j))

        for k in self.e_system.parameter['EName']:
            self.gur_model.addConstr(self.e[k] <= self.e_system.parameter['EMax'], name='%s_%s' % ('e_max0', k))
            self.gur_model.addConstr(self.e[k] >= self.e_system.parameter['EMin'], name='%s_%s' % ('e_min0', k))


    def add_constraint_curve(self):
        for k in self.e_system.parameter['EName']:
            _temp_sum = 0
            for i in range(self.curve.numbers):
                _temp_sum += self.soc[i][k]
            LHS = self.e[k]
            RHS = _temp_sum  # RHS=soc[0][k]+soc[1][k]+soc[2][k]+soc[3][k]+soc[4][k]
            self.gur_model.addConstr(LHS == RHS, name='%s_%s' % ('curve', k))

    def add_constraint_soc(self):
    ### how to constraint for  d_1I_2 <= soc_1 <=d_1I_1?############################
        for k in self.e_system.parameter['EName']:
            for i in range(self.curve.numbers):
                name_num = str(i + 1)
                bench_num = i
                if bench_num == 0:
                    self.gur_model.addConstr(self.soc[bench_num][k] <= float(self.d[bench_num]) * self.I[bench_num][k],
                                    name='%s_%s' % ('soc_' + name_num + '_max', k))
                    self.gur_model.addConstr(float(self.d[bench_num]) * self.I[bench_num + 1][k] <= self.soc[bench_num][k],
                                    name='%s_%s' % ('soc_' + name_num + '_min', k))
                elif bench_num == self.curve.numbers - 1:
                    self.gur_model.addConstr(self.soc[bench_num][k] <= float(self.d[bench_num]) * self.I[bench_num][k],
                                    name='%s_%s' % ('soc_' + name_num + '_max', k))
                    self.gur_model.addConstr(0 <= self.soc[bench_num][k], name='%s_%s' % ('soc_' + name_num + '_min', k))
                else:
                    self.gur_model.addConstr(self.soc[bench_num][k] <= float(self.d[bench_num]) * self.I[bench_num][k],
                                    name='%s_%s' % ('soc_' + name_num + '_max', k))
                    self.gur_model.addConstr(float(self.d[bench_num]) * self.I[bench_num + 1][k] <= self.soc[bench_num][k],
                                    name='%s_%s' % ('soc_' + name_num + '_min', k))

    def add_constraint_I(self):
        for s in range(self.lmp.Nlmp_s):
            for k in self.e_system.parameter['EName']:
                for i in range(self.curve.numbers - 1):
                    name_num = str(i + 1)
                    name_num_next = str(i + 2)
                    bench_num = i
                    self.gur_model.addConstr(self.I[bench_num + 1][k] <= self.I[bench_num][k],
                                    name='%s_%s' % ('I_' + name_num_next + '_' + name_num, k))

    def add_contraint_terminal(self):
        beta = 0.001  # 0.001
        for k in self.e_system.parameter['EName']:
            curr_time = self.curr_model_para.time_period - self.curr_model_para.LAC_bhour
            LHS_1 = self.e[k] - self.e_system.parameter['EEnd']
            RHS_1 = (curr_time) * self.psh_system.parameter['GenMax'] / (
                        self.psh_system.parameter['GenEfficiency'] + beta)  # PSHmax_g[0] / PSHefficiency[0]
            self.gur_model.addConstr(LHS_1 <= RHS_1, name='%s_%s' % ('final_upper', k))
        for k in self.e_system.parameter['EName']:
            curr_time = self.curr_model_para.time_period - self.curr_model_para.LAC_bhour
            LHS_2 = self.e[k] - self.e_system.parameter['EEnd']
            RHS_2 = -(curr_time) * self.psh_system.parameter['PumpMax'] * (
                        self.psh_system.parameter['PumpEfficiency'] - beta)  # PSHmax_p[0] * PSHefficiency[0]
            self.gur_model.addConstr(LHS_2 >= RHS_2, name='%s_%s' % ('final_lower', k))

##the following is for set upt elements of optimiation problems
    def set_up_variable(self):
    #add gen/pump
        self.psh_gen = self.add_var_psh('psh_gen_main')
        self.psh_pump = self.add_var_psh('psh_pump_main')

    # add e
        self.e = self.add_var_e('e_main')

    #add soc and I
        #self.len_var = self.curve.numbers #len(self.curve.point_X)-1
        self.soc = []
        self.I = []
        for i in range(self.curve.numbers):
        #for i in range(self.curve.numbers ):
            name_num = str(i + 1)
            self.soc.append(self.add_var_e('soc_' + name_num))
            self.I.append(self.add_var_I('I_' + name_num))

    #add d
        d = []
        for i in range(self.curve.numbers):
        #for i in range(self.curve.numbers ):
            d.append(self.curve.point_X[i+1] - self.curve.point_X[i])
        self.d = d

        self.gur_model.update()

    def set_up_constraint(self):
    # rolling constraint E_start = E_end +pump + gen
        self.add_constraint_rolling()
    # upper and lower constraint
        self.add_constraint_epsh()
    # curve constraint
        self.add_constraint_curve()
    # constraint for  d_1I_2 <= soc_1 <=d_1I_1?##
        self.add_constraint_soc()
    # constraint for I_1<=I_2<=I_3
        self.add_constraint_I()
    # terminal constraint
        self.add_contraint_terminal()

        self.gur_model.update()

    def set_up_object(self):
        self.profit_max = []
        for j in self.psh_system.parameter['PSHName']:
            self.profit_max.append((self.psh_gen[j] - self.psh_pump[j]) * self.lmp.lmp_scenarios[0][0])
        for k in self.e_system.parameter['EName']:
            for i in range(self.curve.numbers):
                bench_num = i
                #self.profit_max.append(self.curve.point_Y[bench_num] * self.soc[bench_num][k])
                #curve如果是[soc=0, slope=10],[soc=30,slope=20], 从0-30,slope为30
                self.profit_max.append(self.curve.point_Y[bench_num + 1] * self.soc[bench_num ][k])
        print(self.profit_max)
        self.obj = quicksum(self.profit_max)

########################################
########################################
#functions for solve and output results
    def get_optimal_soc(self):
        self.optimal_soc = []
        _temp = list(self.e_system.parameter['EName'])[0]
        for v in [v for v in self.gur_model.getVars() if (_temp in v.Varname and 'soc' in v.Varname)]:
            soc = v.X
            self.optimal_soc.append(soc)
        self.optimal_soc_sum = sum(self.optimal_soc)
        a = self.optimal_soc_sum
        print(a)


    def get_optimal_gen_pump(self):
    #get optimal_psh_gen/pump
        self.optimal_psh_pump = []
        self.optimal_psh_gen = []
        for v in [v for v in self.gur_model.getVars() if 'psh_gen_main' in v.Varname]:
            psh = v.X
            self.optimal_psh_gen.append(psh)
        self.optimal_psh_gen_sum = sum(self.optimal_psh_gen)
        for v in [v for v in self.gur_model.getVars() if 'psh_pump_main' in v.Varname]:
            psh = v.X
            #psh0.append(-psh)
            self.optimal_psh_pump.append(psh)
        self.optimal_psh_pump_sum = sum(self.optimal_psh_pump)
        print(self.optimal_psh_pump_sum)
######################################################
#####################################################
    def get_optimal_profit(self):
    #get optimal profit
        #self.optimal_profit = self.calculate_pts(self.optimal_soc_sum) ##注意这里
        obj = self.gur_model.getObjective() #self.calculate_pts(self.optimal_soc_sum)
        self.optimal_profit = obj.getValue()

    def get_optimal_profit_with_input(self):
    #get optimal profit
        #self.optimal_profit = self.calculate_pts(self.optimal_soc_sum) ##注意这里
        obj = self.gur_model.getObjective() #self.calculate_pts(self.optimal_soc_sum)
        self.optimal_profit_with_input = obj.getValue()

    def get_curr_cost(self):
        #put the soc_sum in, we get the profit
        point_profit = []
        point_price = 0
        for s in range(self.lmp.Nlmp_s):
            p_s = self.lmp.lmp_quantiles[s]
            for j in self.psh_system.parameter['PSHName']:
                point_profit.append((self.optimal_psh_gen_sum - self.optimal_psh_pump_sum) * self.lmp.lmp_scenarios[s][0] * p_s)
                point_price = self.lmp.lmp_scenarios[s][0]
        # for j in self.psh_system.parameter['PSHName']:
        #     point_profit.append((self.psh_gen[j] - self.psh_pump[j]) * self.lmp.lmp_scenarios[0][0])
        self.curr_price = point_price
        self.curr_cost = sum(point_profit)


    def output_optimal(self):
    #output the e for next time
        filename = self.e_system.e_start_folder + '/LAC_Solution_System_SOC_'+ str(self.curr_model_para.LAC_bhour) + '.csv'

        with open(filename, 'w') as wf:
            wf.write('Num_Period,Reservoir_Name,SOC\n')
            _temp = list(self.e_system.parameter['EName'])[0]
            for v in [v for v in self.gur_model.getVars() if (_temp in v.Varname and 'e_main' in v.Varname)]:
                self.optimal_e = v.X
                time = 'T' + str(self.curr_model_para.LAC_bhour)
                name = _temp
                st = time + ',' + '%s,%.1f' % (name, self.optimal_e) + '\n'
                wf.write(st)


    def x_to_soc(self, point_X):
        # change soc_sum to soc_1 + soc_2 + soc_3
        turn_1 = point_X // self.curve.steps
        rest = point_X % self.curve.steps
        point_x_soc = []
        for i in range(self.curve.numbers):
            if turn_1 > 0:
                point_x_soc.append(self.curve.steps)
                turn_1 -= 1
            elif turn_1 == 0:
                point_x_soc.append(rest)
                turn_1 -= 1
            else:
                point_x_soc.append(0)
        return point_x_soc



class RLSetUp(OptModelSetUp):
#psh_system, e_system, lmp, curve, curr_model_para, gur_model
    def __init__(self, psh_system, e_system, lmp, curve, curr_model_para, gur_model):
        self.gur_model = gur_model
        self.psh_system = psh_system
        self.e_system = e_system
        self.lmp = lmp
        self.curve = curve
        self.curr_model_para = curr_model_para

        # self.pre_curve = pre_curve
        # self.pre_lmp = pre_lmp

    def set_up_main(self):
        self.set_up_variable()
        self.set_up_constraint()
        self.set_up_object()

    def solve_model_main(self):
        self.gur_model.setObjective(self.obj, GRB.MAXIMIZE)
        self.gur_model.setParam("MIPGap", 0.0001)
        self.gur_model.optimize()

    def get_optimal_main(self):
        # get optimal soc
        self.get_optimal_soc()
        self.get_optimal_gen_pump()
        self.get_optimal_profit()
        self.get_curr_cost()
        self.output_optimal()




##important function
    def optimization_model(self):
        self.set_up_main()
        self.solve_model_main()
        #deal with optimal solution: store and output
        self.get_optimal_main()
        ###update curve and output curve

        #self.get_new_curve_main()



    def optimization_model_with_input(self):#SOC_initial
        self.set_up_main()
        #self.psh_system.parameter['EStart'] = 0#SOC_initial
        self.solve_model_main()
        #deal with optimal solution: store and output
        #self.get_optimal_main()
        #self.get_optimal_soc()
        #self.get_optimal_gen_pump()
        self.get_optimal_profit_with_input()
        #self.output_optimal()

















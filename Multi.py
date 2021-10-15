import multiprocessing as mp
from multiprocessing import *
import gurobipy as grb
from gurobipy import * #GRB
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from SystemSetUp import *
from CurrModelPara import *
from Curve import *



class MulOptModelSetUp():

    # def __init__(self, psh_system, e_system, lmp, curve, curr_model_para, gur_model):
        # self.gur_model = gur_model
        # self.psh_system = psh_system
        # self.e_system = e_system
        # self.lmp = lmp
        # self.curve = curve
        # self.curr_model_para = curr_model_para
    def __init__(self):
        self.gur_model = None
        self.psh_system = None
        self.e_system = None
        self.lmp = None
        self.curve = None
        self.curr_model_para = None
        # self.pre_curve = pre_curve
        # self.pre_lmp = pre_lmp
        # self.alpha = 0.8  # 0.2
        # self.date = 'March 07 2019'
        # self.LAC_last_windows = 0  # 1#0
        # self.probabilistic = 1  # 0#1
        # self.RT_DA = 1  # 0#1
        # self.curr_day = 1
        # self.curr_scenario = 2
        # self.current_stage = 'training_500'

########################################
# funtions for set up
    def add_var_e(self, var_name):
        return self.gur_model.addVars(self.e_system.parameter['EName'], ub=float('inf'),lb=-float('inf'), vtype="C", name=var_name)

    def add_var_I(self, var_name):
        return self.gur_model.addVars(self.e_system.parameter['EName'], vtype="B", name=var_name)

    def add_var_psh(self, var_name):
        return self.gur_model.addVars(self.psh_system.parameter['PSHName'], ub=float('inf'),lb=-float('inf'),vtype="C",name=var_name)

    def add_constraint_week_rolling(self):
        ## SOC0: e_0=E_start; loop from 0 to 22; e_1=e_0+psh1;....e_23=e_22+psh_23; when loop to 22; directly add e_23=E_end
        for k in self.e_system.parameter['EName']:
            print('Estart:', float(self.e_system.parameter['EStart']))
            LHS = self.e[k] + grb.quicksum(self.psh_gen[j] / self.psh_system.parameter['GenEfficiency'] for j in self.psh_system.parameter['PSHName']) \
                                          - grb.quicksum(self.psh_pump[j] * self.psh_system.parameter['PumpEfficiency'] for j in self.psh_system.parameter['PSHName'])
            RHS = self.e_system.parameter['EStart']
            print(LHS)
            ###if we calculate the first one, we use 'SOC0', and the last we use 'End'; or we choose the SOC0 to "beginning", at the same time the last we use 'SOC'.
            self.gur_model.addConstr(LHS == RHS, name='%s_%s' % ('SOC0', k))


    def add_constraint_week_epsh(self):
        for j in self.psh_system.parameter['PSHName']:  # all are lists
            self.gur_model.addConstr(self.psh_gen[j] <= self.psh_system.parameter['GenMax'], name='%s_%s' % ('psh_gen_max0', j))
            self.gur_model.addConstr(self.psh_gen[j] >= self.psh_system.parameter['GenMin'], name='%s_%s' % ('psh_gen_min0', j))
            self.gur_model.addConstr(self.psh_pump[j] <= self.psh_system.parameter['PumpMax'], name='%s_%s' % ('psh_pump_max0', j))
            self.gur_model.addConstr(self.psh_pump[j] >= self.psh_system.parameter['PumpMin'], name='%s_%s' % ('psh_pump_min0', j))

            for i in range(self.one_day_period):
                self.gur_model.addConstr(self.psh_gen_prev[i][j] <= self.psh_system.parameter['GenMax'], name='%s_%s' % ('psh_gen_max'+str(i),j))
                self.gur_model.addConstr(self.psh_gen_prev[i][j] >= self.psh_system.parameter['GenMin'], name='%s_%s' % ('psh_gen_min'+str(i),j))
                self.gur_model.addConstr(self.psh_pump_prev[i][j] <= self.psh_system.parameter['PumpMax'], name='%s_%s' % ('psh_pump_max'+str(i),j))
                self.gur_model.addConstr(self.psh_pump_prev[i][j] >= self.psh_system.parameter['PumpMin'], name='%s_%s' % ('psh_pump_min'+str(i),j))


        for k in self.e_system.parameter['EName']:
            self.gur_model.addConstr(self.e[k] <= self.e_system.parameter['EMax'], name='%s_%s' % ('e_max0', k))
            self.gur_model.addConstr(self.e[k] >= self.e_system.parameter['EMin'], name='%s_%s' % ('e_min0', k))

            for i in range(self.one_day_period):
                self.gur_model.addConstr(self.e_prev[i][k] <= self.e_system.parameter['EMax'], name='%s_%s' % ('e_max'+ str(i), k))
                self.gur_model.addConstr(self.e_prev[i][k] >= self.e_system.parameter['EMin'], name='%s_%s' % ('e_min'+ str(i), k))


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

    def add_constraint_terminal(self):
        beta = 0.001
        for k in self.e_system.parameter['EName']:
            curr_day = self.curr_model_para.day_period  - self.curr_model_para.curr_day
            LHS_1 = self.e[k] - self.e_system.parameter['EEnd']
            RHS_1 = (curr_day) * self.psh_system.parameter['GenMax'] /(self.psh_system.parameter['GenEfficiency']+beta) # PSHmax_g[0] / PSHefficiency[0]
            self.gur_model.addConstr(LHS_1 <= RHS_1, name='%s_%s' % ('final_upper', k))
        for k in self.e_system.parameter['EName']:
            curr_day = self.curr_model_para.day_period - self.curr_model_para.curr_day
            LHS_2 = self.e[k] - self.e_system.parameter['EEnd']
            RHS_2 = -(curr_day) * self.psh_system.parameter['PumpMax'] * (self.psh_system.parameter['PumpEfficiency']- beta) #PSHmax_p[0] * PSHefficiency[0]
            self.gur_model.addConstr(LHS_2 >= RHS_2, name='%s_%s' % ('final_lower', k))

# the following is for set upt elements of optimization problems

    def set_up_constraint(self):
    # rolling constraint E_start = E_end +pump + gen
        self.add_constraint_week_rolling()
    # upper and lower constraint
        self.add_constraint_week_epsh()
    # curve constraint
        self.add_constraint_curve()
    # constraint for  d_1I_2 <= soc_1 <=d_1I_1?##
        self.add_constraint_soc()
    # constraint for I_1<=I_2<=I_3
        self.add_constraint_I()
    # terminal constraint
        self.add_constraint_terminal()

        self.gur_model.update()

    def set_up_week_variable(self):
    #add gen/pump
        self.one_day_period = 23

        self.psh_gen_prev = []
        self.psh_pump_prev  = []

        self.e_prev = []
        for i in range(self.one_day_period):
            name_num = str(i + 1)
            self.psh_gen_prev.append((self.add_var_psh('psh_gen_' + name_num)))
            self.psh_pump_prev.append((self.add_var_psh('psh_pump_' + name_num)))
            self.e_prev.append((self.add_var_e('e_' + name_num)))

        self.psh_gen = self.add_var_psh('psh_gen_main')
        self.psh_pump = self.add_var_psh('psh_pump_main')
        self.e = self.add_var_e('e_main')

    # add e


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

    def set_up_week_object(self):
        self.profit_max = []
        for j in self.psh_system.parameter['PSHName']:
            self.profit_max.append((self.psh_gen[j] - self.psh_pump[j]) * self.lmp.lmp_scenarios)
            for i in range(self.one_day_period):
                self.profit_max.append((self.psh_gen_prev[i][j] - self.psh_pump_prev[i][j]) * self.lmp.lmp_scenarios_prev[i])

        for k in self.e_system.parameter['EName']:
            for i in range(self.curve.numbers):
                bench_num = i
                self.profit_max.append(self.curve.point_Y[bench_num + 1] * self.soc[bench_num][k])
        print(self.profit_max)


        self.obj = quicksum(self.profit_max)




########################################
########################################
# functions for solve and output results
    def get_optimal_soc(self):

        self.optimal_soc = []
        _temp = list(self.e_system.parameter['EName'])[0]
        for v in [v for v in self.gur_model.getVars() if (_temp in v.Varname and 'soc' in v.Varname)]:
            soc = v.X
            self.optimal_soc.append(soc)
        self.optimal_soc_sum = sum(self.optimal_soc)
        #a = self.optimal_soc_sum
        #print(a)

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

    def get_optimal_profit(self):
    #get optimal profit
        #self.optimal_profit = self.calculate_pts(self.optimal_soc_sum) ##注意这里

        obj = self.gur_model.getObjective() #self.calculate_pts(self.optimal_soc_sum)
        #self.optimal_profit = obj.getValue()
        return obj.getValue()

    def get_curr_cost(self):
        #put the soc_sum in, we get the profit
        point_profit = []

        point_profit.append((self.optimal_psh_gen_sum - self.optimal_psh_pump_sum) * self.lmp.lmp_scenarios)
        # for j in self.psh_system.parameter['PSHName']:
        #     point_profit.append((self.psh_gen[j] - self.psh_pump[j]) * self.lmp.lmp_scenarios[0][0])

        self.curr_cost = sum(point_profit)


    def output_optimal(self):
    #output the e for next time
        filename = self.e_system.e_start_folder + '/LAC_Solution_System_SOC_'+ str(self.curr_model_para.curr_day) + '.csv'

        with open(filename, 'w') as wf:
            wf.write('Num_Period,Reservoir_Name,SOC\n')
            _temp = list(self.e_system.parameter['EName'])[0]
            for v in [v for v in self.gur_model.getVars() if (_temp in v.Varname and 'e_main' in v.Varname)]:
                self.optimal_e = v.X
                time = 'T' + str(self.curr_model_para.curr_day)
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



#class MulRLSetUp(OptModelSetUp):
class MulRLSetUp(MulOptModelSetUp):
# #psh_system, e_system, lmp, curve, curr_model_para, gur_model

    def set_up_main(self):
        self.set_up_week_variable()
        self.set_up_constraint()
        self.set_up_week_object()


    def solve_model_main(self):
        self.gur_model.setObjective(self.obj, GRB.MAXIMIZE)
        self.gur_model.setParam("MIPGap", 0.0001)
        self.gur_model.optimize()

    def get_optimal_main(self):
        # get optimal soc
        #self.get_optimal_soc()
        #self.get_optimal_gen_pump()
        self.get_optimal_profit()
        #self.get_curr_cost()
        #self.output_optimal()

        ########################################

    def SetUpMain(self, initial_soc):
        # self.alpha = 0.8  # 0.2
        # self.date = 'March 07 2019'
        # self.LAC_last_windows = 0  # 1#0
        # self.probabilistic = 1  # 0#1
        # self.RT_DA = 1  # 0#1
        # self.curr_day = 1
        # self.curr_scenario = 2
        # self.current_stage = 'training_500'
        self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                             self.curr_day,
                                             self.curr_scenario, self.current_stage, self.day_period)
        # LAC_last_windows,  probabilistic, RT_DA, date, curr_day, scenario

        self.psh_system = PshSystem(self.curr_model_para)
        self.psh_system.set_up_parameter()

        self.e_system = ESystem(self.curr_model_para)
        self.e_system.set_up_parameter()
        self.e_system.parameter['EStart'] = initial_soc

        if self.curr_day != self.curr_model_para.day_period - 1:
            # lmp, time = t+1, scenario= n
            self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                                 self.curr_day + 1,
                                                 self.curr_scenario, self.current_stage, self.day_period)
            self.lmp = LMP(self.curr_model_para)
            self.lmp.seven_set_up_parameter()
            # curve, time = t+1, scenario= n-1
            self.curve = Curve(100, 0, 3000, self.day_period)
            # self.curve.input_curve(self.curr_day + 1, self.curr_scenario - 1)
            self.curve.input_curve(self.curr_day + 1, self.curr_scenario - 1)
        elif self.curr_day == self.curr_model_para.day_period - 1:
            self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                                 self.curr_day,
                                                 self.curr_scenario, self.current_stage, self.day_period)
            self.lmp = LMP(self.curr_model_para)
            #self.lmp.set_up_parameter()
            self.lmp.seven_set_up_parameter()


            self.curve = Curve(100, 0, 3000, self.day_period)
            self.curve.input_curve(self.curr_day, self.curr_scenario - 1)

        self.gur_model = Model('DAMarket')


    def MainMultWithInput(self, initial_soc):#SOC_initial
        self.SetUpMain(initial_soc)
        #with grb.Env() as env, grb.Model(env=env) as self.gur_model:
             #在这里才用到
        self.set_up_main()
        self.solve_model_main()
        #self.get_optimal_main()
        _temp = self.get_optimal_profit()
        #return self.optimal_profit
        return _temp

##the most most important function

    def CalOpt(self, initial_input):
        #if __name__ == '__main__':

        with mp.Pool() as pool:
            _temp = pool.map(self.MainMultWithInput, initial_input)
        self.optimal_profit_list = _temp








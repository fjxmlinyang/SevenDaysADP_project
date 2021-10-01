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

# class CurrModelPara():
#     def __init__(self, LAC_last_windows, probabilistic, RT_DA, date, LAC_bhour, scenario, current_stage):
#
#         # set the length of the rolling window
#         # LAC_window = 1
#
#         # indicate if the current window is the last window, default as 0
#         # LAC_last_windows = 0
#
#         # 0:apply the deterministic forecast, 1:apply the probabilistic forecast
#         #probabilistic = 0
#
#         # read time periods
#
#         self.LAC_last_windows = LAC_last_windows
#         self.RT_DA = RT_DA
#         self.probabilistic = probabilistic
#         self.date = date
#         self.LAC_bhour = LAC_bhour
#         self.scenario = scenario
#         self.current_stage = current_stage
#
#
# class Curve(object):
#     def __init__(self, numbers, lo_bd, up_bd):
#         '''
#         :param numbers: numbers of the segments
#         :param lo_bd: the lower bound of the curve
#         :param up_bd: the upper bound of the curve
#         '''
#         self.numbers = numbers
#         self.up_bd = up_bd
#         self.lo_bd = lo_bd
#         self.steps = (up_bd - lo_bd) // numbers
#         self.filename_all = './Output_Curve'
#         self.seg_initial()
#         self.curve_initial()
#         self.output_initial_curve()
#
#     def seg_initial(self):
#         '''
#         :return: curve with initial value
#         '''
#         segments = []
#         for i in range(self.lo_bd, self.up_bd + self.steps, self.steps):
#             curr_step = i // self.steps
#             if i == self.lo_bd:
#                 value = 50
#                 self.intial_slope_set = value
#             else:
#                 value = value - 0.02 * self.steps  # /10
#                 # value = 50 - curr_step *0.4
#             # value = (100 - 2*i // self.steps)
#             segments.append([i, value])
#         self.segments = segments
#
#     def seg_update(self, point_1, point_2):
#         '''
#         :param point_1:  update point one
#         :param point_2:  update point two
#         :return:
#         '''
#         point_1_x = point_1[0]
#         point_1_y = point_1[1]
#         point_2_x = point_2[0]
#         point_2_y = point_2[1]
#         for i in range(self.numbers + 1):
#             curr = self.segments[i]
#             curr_x = curr[0]
#             curr_y = curr[1]
#             if curr_x <= point_1_x and curr_y <= point_1_y:
#                 self.segments[i][1] = point_1_y
#             elif curr_x >= point_2_x and curr_y >= point_2_y:
#                 self.segments[i][1] = point_2_y
#         self.curve_initial()  # 需要把point_X and point_Y更新下
#         print(self.segments)
#
#     def curve_initial(self):
#         '''
#         :return: list the x value and y value of the curve
#         '''
#         df = pd.DataFrame(self.segments, columns=['x', 'y'])
#         self.curve_df = df
#         self.point_X = self.curve_df['x'].to_list()
#         self.point_Y = self.curve_df['y'].to_list()
#
#     def show_curve(self):
#         '''
#         :return: print the curve
#         '''
#         sns.set_theme(style="darkgrid")
#         sns.lineplot(x='x', y='y', data=self.curve_df)
#         plt.show()
#
#     def curve_update(self, new_curve_Y, point_1, point_2):
#         '''
#         :param new_curve_Y: the curve for update
#         :param point_1: update point 1
#         :param point_2: update point 2
#         :return: update curve
#         '''
#         for i in range(len(new_curve_Y)):
#             value = new_curve_Y[i]
#             self.segments[i][1] = value
#         self.seg_update(point_1, point_2)
#
#     def input_curve(self, time, scenario):
#         '''
#         :param time:  hours
#         :param scenario: scenarios
#         :return: read the curve in folder
#         '''
#         _str = str(time)
#         filename = self.filename_all + '/Curve_' + 'time_' + _str + '_scenario_' + str(scenario) + '.csv'
#         df = pd.read_csv(filename)
#         self.segments = df.values.tolist()
#         self.curve_initial()  # !!!别忘了
#         print(self.segments)
#
#     def output_initial_curve(self):
#         # output the initial curve
#         for curr_time in range(24):
#             _str = str(curr_time)
#             scenario = 0
#             filename = self.filename_all + '/Curve_' + 'time_' + _str + '_scenario_' + str(scenario) + '.csv'
#             df = pd.DataFrame(self.segments, columns=['soc_segment', 'slope'])
#             df.to_csv(filename, index=False, header=True)
#
#
# class System():
#     def __init__(self, curr_model):
#         self.curr_model = curr_model
#         self.Input_folder_parent = None
#         self.filename = None
#         self.Input_folder = None
#         self.Output_folder = None
#         self.parameter = {}
#         # this is for easy
#         if curr_model.current_stage == 'training_50':
#             self.Input_all_total = './Input_Curve'
#         if curr_model.current_stage == 'training_500':
#             self.Input_all_total = './Input_bootstrap'
#         if curr_model.current_stage == 'test':
#             self.Input_all_total = './Input_test'
#
#     def input_parameter(self, paranameter_name, in_model_name):
#         Data = pd.read_csv(self.filename)
#         df = pd.DataFrame(Data)
#         # ret = list(df[paranameter_name])
#         ret = df[paranameter_name]
#         self.parameter[in_model_name] = ret  # [0]
#
#
# class PshSystem(System):
#
#     def set_up_parameter(self):
#         ##这个是给标量 #how to input one by one?
#         self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
#         self.Input_folder = self.Input_folder_parent + '/' + self.curr_model.date
#         self.filename = self.Input_folder + '/PSH.csv'
#         self.input_parameter('GenMin', 'GenMin')
#         self.input_parameter('GenMax', 'GenMax')
#         self.input_parameter('PumpMin', 'PumpMin')
#         self.input_parameter('PumpMax', 'PumpMax')
#         self.input_parameter('Cost', 'Cost')
#         self.input_parameter('Efficiency', 'GenEfficiency')
#         self.input_parameter('Efficiency', 'PumpEfficiency')
#         self.input_parameter('Name', 'PSHName')
#
#         self.Input_folder = None
#         self.filename = None
#
#
# class ESystem(System):
#
#     def set_up_parameter(self):
#         ##这个是给标量 #how to input one by one?
#         self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
#         self.Input_folder = self.Input_folder_parent + '/' + self.curr_model.date
#         self.filename = self.Input_folder + '/Reservoir.csv'
#         self.input_parameter('Min', 'EMin')
#         self.input_parameter('Max', 'EMax')
#         self.input_parameter('Name', 'EName')
#         self.input_parameter('End', 'EEnd')
#
#         self.Output_folder = './Output_Curve'
#         # here are for rolling model
#         # here we can set the benchmark?
#         if self.curr_model.LAC_bhour == 0:
#             self.input_parameter('Start', 'EStart')
#             self.e_start_folder = self.Output_folder
#         elif self.curr_model.LAC_last_windows:
#             self.filename = self.Output_folder + '/LAC_Solution_System_SOC_' + str(
#                 self.curr_model.LAC_bhour - 1) + '.csv'
#             self.input_parameter('SOC', 'EStart')
#             self.e_start_folder = self.Output_folder
#         else:
#             self.filename = self.Output_folder + '/LAC_Solution_System_SOC_' + str(
#                 self.curr_model.LAC_bhour - 1) + '.csv'
#             self.input_parameter('SOC', 'EStart')
#             self.e_start_folder = self.Output_folder
#         self.Input_folder = None
#         self.filename = None
#         self.Output_folder = None
#
#
# class LMP(System):
#
#     def set_up_parameter(self):
#         self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
#         self.Input_folder = self.Input_folder_parent + '/' + self.curr_model.date
#         if self.curr_model.LAC_last_windows:
#             # filename = Input_folder + '\LMP_Hindsight' + '.csv'
#             # self.filename = self.Input_folder + '/prd_dataframe_wlen_24_'+ self.curr_model.date + '.csv'
#             self.filename = self.Input_folder + '/prd_dataframe_wlen_' + str(
#                 24 - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '.csv'
#         else:
#             # filename = Input_folder+'\LMP_Scenarios_' + 'T' + str(LAC_bhour) +'_DA'+ '.csv'
#             if self.curr_model.probabilistic and self.Input_all_total == './Input_bootstrap':
#                 self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(
#                     24 - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '_550' + '.csv'
#             elif self.curr_model.probabilistic and self.Input_all_total == './Input_test':
#                 self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(
#                     24 - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '_550' + '.csv'
#             elif self.curr_model.probabilistic and self.Input_all_total == './Input_Curve':
#                 self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(
#                     24 - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '_50' + '.csv'
#             else:
#                 self.filename = self.Input_folder + '/prd_dataframe_wlen_' + str(
#                     24 - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '.csv'
#
#         Data = pd.read_csv(self.filename)
#         df = pd.DataFrame(Data)
#         Column_name = list(Data.columns)
#         self.lmp_quantiles = []
#         self.lmp_scenarios = []
#         # DA_lmp=[]???
#         if self.curr_model.LAC_last_windows:
#             self.Nlmp_s = 1
#             # probability of each scenario is evenly distributed
#             self.lmp_quantiles.append(1.0 / self.Nlmp_s)
#             if self.curr_model.RT_DA == 1:
#                 self.lmp_scenarios.append(list(df['RT_LMP']))
#             else:
#                 self.lmp_scenarios.append(list(df['DA_LMP']))
#         else:
#             if self.curr_model.probabilistic:
#                 self.Nlmp_s = len(Column_name)
#                 for i in range(self.Nlmp_s):
#                     # probability of each scenario is evenly distributed
#                     self.lmp_quantiles.append(1.0 / self.Nlmp_s)
#                     ##only change here!!!
#                     self.lmp_scenarios.append(list(df[Column_name[self.curr_model.scenario]]))
#             else:
#                 # for deterministic forecast, there is a single scenario
#                 self.Nlmp_s = 1
#                 self.lmp_quantiles.append(1.0 / self.Nlmp_s)
#                 # deterministic forecast is the single point prediction
#                 self.lmp_scenarios.append(list(df['prd']))
#
#         self.Input_folder = None
#         self.filename = None
#         self.Output_folder = None
#
#
#


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
        # self.curr_time = 1
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

    def add_constraint_terminal(self):
        beta = 0.001
        for k in self.e_system.parameter['EName']:
            curr_time = self.curr_model_para.time_period  - self.curr_model_para.LAC_bhour
            LHS_1 = self.e[k] - self.e_system.parameter['EEnd']
            RHS_1 = (curr_time   ) * self.psh_system.parameter['GenMax'] /(self.psh_system.parameter['GenEfficiency']+beta) # PSHmax_g[0] / PSHefficiency[0]
            self.gur_model.addConstr(LHS_1 <= RHS_1, name='%s_%s' % ('final_upper', k))
        for k in self.e_system.parameter['EName']:
            curr_time = self.curr_model_para.time_period - self.curr_model_para.LAC_bhour
            LHS_2 = self.e[k] - self.e_system.parameter['EEnd']
            RHS_2 = -(curr_time  ) * self.psh_system.parameter['PumpMax'] * (self.psh_system.parameter['PumpEfficiency']- beta) #PSHmax_p[0] * PSHefficiency[0]
            self.gur_model.addConstr(LHS_2 >= RHS_2, name='%s_%s' % ('final_lower', k))

# the following is for set upt elements of optimization problems

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
        self.add_constraint_terminal()

        self.gur_model.update()

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
        for s in range(self.lmp.Nlmp_s):
            p_s = self.lmp.lmp_quantiles[s]
            for j in self.psh_system.parameter['PSHName']:
                point_profit.append((self.optimal_psh_gen_sum - self.optimal_psh_pump_sum) * self.lmp.lmp_scenarios[s][0] * p_s)
        # for j in self.psh_system.parameter['PSHName']:
        #     point_profit.append((self.psh_gen[j] - self.psh_pump[j]) * self.lmp.lmp_scenarios[0][0])

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



#class MulRLSetUp(OptModelSetUp):
class MulRLSetUp(MulOptModelSetUp):
# #psh_system, e_system, lmp, curve, curr_model_para, gur_model

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
        # self.curr_time = 1
        # self.curr_scenario = 2
        # self.current_stage = 'training_500'
        self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                             self.curr_time,
                                             self.curr_scenario, self.current_stage, self.time_period)
        # LAC_last_windows,  probabilistic, RT_DA, date, LAC_bhour, scenario

        self.psh_system = PshSystem(self.curr_model_para)
        self.psh_system.set_up_parameter()

        self.e_system = ESystem(self.curr_model_para)
        self.e_system.set_up_parameter()
        self.e_system.parameter['EStart'] = initial_soc

        if self.curr_time != self.curr_model_para.time_period - 1:
            # lmp, time = t+1, scenario= n
            self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                                 self.curr_time + 1,
                                                 self.curr_scenario, self.current_stage, self.time_period)
            self.lmp = LMP(self.curr_model_para)
            self.lmp.set_up_parameter()
            # curve, time = t+1, scenario= n-1
            self.curve = Curve(100, 0, 3000, self.time_period)
            # self.curve.input_curve(self.curr_time + 1, self.curr_scenario - 1)
            self.curve.input_curve(self.curr_time + 1, self.curr_scenario - 1)
        elif self.curr_time == self.curr_model_para.time_period - 1:
            self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                                 self.curr_time,
                                                 self.curr_scenario, self.current_stage, self.time_period)
            self.lmp = LMP(self.curr_model_para)
            self.lmp.set_up_parameter()

            self.curve = Curve(100, 0, 3000, self.time_period)
            self.curve.input_curve(self.curr_time, self.curr_scenario - 1)

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








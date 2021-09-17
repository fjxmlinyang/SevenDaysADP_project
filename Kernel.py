import pandas as pd
from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import numpy as np
from ModelSetUp import *
from CurrModelPara import *
from Curve import *
#from Main_cal_opt import find_optimal_value
from multiprocess import *


class RL_Kernel():
    def __init__(self):
        #self.reward = None
        #self.value = None
        #self.action = None
        self.alpha = None #0, 0.2 , 0.5, 0.8, 1
        self.date = None#'March 07 2019'
        self.LAC_last_windows = 1 #0#1 #必须是1才可以是DA的price
        self.probabilistic = 0 #1#0
        self.RT_DA = 0 #1#0
        self.curr_time = None
        self.curr_scenario = None
        self.current_stage ='training_50' #'training_500'
        #如果我们要用repetitive DA， 我们需要LAC_last_windows = 0， probabilitsit = 1, DA = 0?
        self.time_period = 23 #24? #24-1?

    def main_function(self):
        time_1 = time.time()
        self.Curr_Scenario_Cost_Total = []
        if self.date =='March 07 2019':
            self.start = 1+1
            self.end = 600+1
        if self.date =='April 01 2019':
            self.start = 600+1
            self.end = 1200+1
        if self.date =='April 15 2019':
            self.start = 1200+1
            self.end = 1800+1
        if self.date =='April 22 2019':
            self.start = 1800+1
            self.end = 2400+1
        for curr_scenario in range(self.start, self.end):
            self.Curr_Scenario_Price_Total = []
            self.PSH_Results = []
            self.SOC_Results = []
            self.curr_scenario_cost_total = 0
            self.curr_price_total = []
            for i in range(self.time_period):
                self.curr_time = i
                self.curr_scenario = curr_scenario
                self.calculate_optimal_soc()
                self.get_final_curve_main()
                self.output_psh_soc()
            self.output_psh_soc_main()
        self.output_curr_cost()
        time_2 = time.time()
        print('one iteration time is', time_2-time_1)


    def output_curr_cost(self):
        # output the psh and soc
        filename = './Output_Curve' + '/PSH_Profitmax_Rolling_Results_' + 'total_' + str(
            self.curr_scenario) + '_' + self.date +'_alpha_' + str(int(self.alpha*10)) +'.csv'
        self.df_total.to_csv(filename)

        # output curr_cost
        filename = './Output_Curve' + '/Current_Cost_Total_Results_' + str(
            self.curr_scenario) + '_' + self.date +'_alpha_' + str(int(self.alpha*10)) + '.csv'
        self.df = pd.DataFrame({'Curr_Scenario_Cost_Total': self.Curr_Scenario_Cost_Total})
        self.df.to_csv(filename)

    def output_psh_soc_main(self):
        # add the last one

        filename = './Output_Curve' + '/PSH_Profitmax_Rolling_Results_' + str(
            self.curr_scenario) + '_' + self.date + '.csv'
        if self.SOC_Results[-1] - self.e_system.parameter['EEnd'][0] > 0.1:
            self.PSH_Results.append(
                (self.SOC_Results[-1] - self.e_system.parameter['EEnd'][0]) * self.psh_system.parameter['GenEfficiency'][0])
        else:
            self.PSH_Results.append(
                (self.SOC_Results[-1] - self.e_system.parameter['EEnd'][0]) / self.psh_system.parameter['PumpEfficiency'][0])

        self.SOC_Results.append(self.e_system.parameter['EEnd'][0])

        # return price for one scenario
        # add last price here, then what information I need? scenario, and read the price
        filename = './Input_Curve/PSH-Rolling Window' + '/'+ self.date + '/DA_lmp_Scenarios_wlen_' + str(1) + '_'+ self.date+'_50' + '.csv'
        Data = pd.read_csv(filename)
        df = pd.DataFrame(Data)
        number = (self.curr_scenario) % 50 - 1
        #'V' + str(self.curr_scenario % 50)
        cur_list = df.iloc[:, number]

        self.curr_price_total.append(cur_list[0])
        self.curr_scenario_cost_total += cur_list[0] * self.PSH_Results[-1]



        self.Curr_Scenario_Price_Total.append(self.curr_price_total)


        self.df = pd.DataFrame(
            {'Price_Results_' + str(self.curr_scenario): self.Curr_Scenario_Price_Total[0], 'SOC_Results_' + str(self.curr_scenario): self.SOC_Results, 'PSH_Results_' + str(self.curr_scenario): self.PSH_Results})
        # df = pd.DataFrame({'PSH_Results_' + str(curr_scenario): PSH_Results})
        # df.to_csv(filename)
        if self.curr_scenario == self.start:
            self.df_total = self.df
        else:
            self.df_total = pd.concat([self.df_total, self.df], axis=1)




        ##calculate total cost
        self.Curr_Scenario_Cost_Total.append(self.curr_scenario_cost_total)

    def output_psh_soc(self):
        self.SOC_Results.append(self.curr_model.optimal_soc_sum)
        if self.curr_model.optimal_psh_gen_sum > 1:#0.1:
            self.PSH_Results.append(self.curr_model.optimal_psh_gen_sum)
        else:
            self.PSH_Results.append(-self.curr_model.optimal_psh_pump_sum)

        ##output curr cost #这里就全部加起来了
        self.curr_scenario_cost_total += self.curr_model.curr_cost
        #
        self.curr_price_total.append(self.curr_model.curr_price)


    def calculate_optimal_soc(self):
        self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                             self.curr_time, self.curr_scenario, self.current_stage, self.time_period)        # LAC_last_windows,  probabilistic, RT_DA, date, LAC_bhour, scenario

        print('##############################' + 'scenario = ' + str(self.curr_scenario) + ', and curr_time = ' + str(self.curr_time) + '######################################')



        print('################################## psh_system set up ##################################')
        self.psh_system = PshSystem(self.curr_model_para)
        self.psh_system.set_up_parameter()
        print(self.psh_system.parameter)

        print('################################## e_system set up ##################################')
        self.e_system = ESystem(self.curr_model_para)
        self.e_system.set_up_parameter()
        print(self.e_system.parameter)

        print('################################## lmp_system set up ##################################')
        self.lmp = LMP(self.curr_model_para)
        self.lmp.set_up_parameter()
        #print(self.lmp.date)
        print('lmp_quantiles=', self.lmp.lmp_quantiles)
        print('lmp_scenarios=', self.lmp.lmp_scenarios)
        print('lmp_Nlmp_s=', self.lmp.Nlmp_s)

        print('################################## curve set up ##################################')
        self.old_curve = Curve(100, 0, 3000, self.time_period)


        ####不同的开始，不同的curve
        if self.curr_scenario == 1 and self.curr_time == 0:
            self.old_curve.output_initial_curve()

        if self.LAC_last_windows == 0 and self.probabilistic == 1 and self.RT_DA == 0 and self.curr_scenario == 1:
            last_scenario = 10000
            self.old_curve.input_tuned_initial_curve(last_scenario)
        self.old_curve.input_curve(self.curr_time, self.curr_scenario - 1)
        print(self.old_curve.segments)

        print('################################## ADP training model set up ##################################')
        model_1 = Model('DAMarket')
        self.curr_model = RLSetUp(self.psh_system, self.e_system, self.lmp, self.old_curve, self.curr_model_para, model_1)
        self.curr_model.optimization_model()
        self.optimal_soc_sum = self.curr_model.optimal_soc_sum
        self.optimal_psh_gen_sum = self.curr_model.optimal_psh_gen_sum
        self.optimal_psh_pump_sum = self.curr_model.optimal_psh_pump_sum

        print(self.curr_model.optimal_soc_sum)


    def calculate_new_soc(self, initial_soc):
        pre_model = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date, self.curr_time,
                                  self.curr_scenario, self.current_stage, self.time_period)
        # LAC_last_windows,  probabilistic, RT_DA, date, LAC_bhour, scenario

        psh_system_2 = PshSystem(pre_model)
        psh_system_2.set_up_parameter()


        e_system_2 = ESystem(pre_model)
        e_system_2.set_up_parameter()
        e_system_2.parameter['EStart'] = initial_soc
        #print('e_system_2.parameter is ' + str(e_system_2.parameter))
        if self.curr_time != self.time_period - 1:
            # lmp, time = t+1, scenario= n
            self.prev_model = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                            self.curr_time + 1,
                                            self.curr_scenario, self.current_stage, self.time_period)
            self.prev_lmp = LMP(self.prev_model)
            self.prev_lmp.set_up_parameter()
            # curve, time = t+1, scenario= n-1
            self.pre_curve = Curve(100, 0, 3000, self.time_period)
            self.pre_curve.input_curve(self.curr_time + 1, self.curr_scenario - 1)
        elif self.curr_time == self.time_period - 1:
            self.prev_model = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date,
                                            self.curr_time,
                                            self.curr_scenario, self.current_stage, self.time_period)
            self.prev_lmp = LMP(self.prev_model)
            self.prev_lmp.set_up_parameter()

            self.pre_curve = Curve(100, 0, 3000, self.time_period)
            self.pre_curve.input_curve(self.curr_time, self.curr_scenario - 1)

        model_1 = Model('DAMarket')
        #ADP_train_model_para = pre_model
        a = self.prev_lmp.lmp_scenarios
        print(a)
        b = self.pre_curve.point_Y
        print(b)

        pre_model = RLSetUp(psh_system_2, e_system_2, self.prev_lmp, self.pre_curve, pre_model, model_1)
        pre_model.optimization_model_with_input()
        rt = pre_model.optimal_profit_with_input
        return rt



#after we get the current self.optimal_profit and self.optimal_soc_sum, we have to update the curve

    def get_final_curve_main(self):
        #print('test')
        self.get_new_curve_step_0()
        # new curve: self.new_curve_slope

        self.get_new_curve_step_1_multi()  # 基于此次最优解的model
        print(self.curve.segments)
        self.get_new_curve_step_2_curve_comb()  # (1-\alpha)*old_curve + \alpha*old_curve
        self.get_new_curve_step_3_two_pts()  # update the new curve with the two new points
        print(self.second_curve_slope)

        # new points: self.update_point_1 and self.update_point_2
        self.curve.curve_update(self.new_curve_slope, self.update_point_1, self.update_point_2)
        print(self.curve.segments)
        self.output_curve()
        self.output_curve_sum()

    def get_new_curve_step_0(self):
        # how can we get each new curve_point_X
        self.curve = self.old_curve
        self.second_curve_soc = self.curve.point_X

        # get new curve_profit
        self.second_curve_profit = []

        # make sure its terminal soc works
        self.check_soc_curve = []

    def get_new_curve_step_1_multi(self):
        beta = 0.001
        #让无法到的点设置成为着-10000
        for value in self.second_curve_soc:
            distance = value - float(self.e_system.parameter['EEnd'])
            left_cod = distance <= 0 and (abs(distance) < (self.time_period - self.curr_time) * float(
                self.psh_system.parameter['PumpMax']) * (float(self.psh_system.parameter['PumpEfficiency']) - beta))
            right_cod = distance > 0 and (abs(distance) < (self.time_period - self.curr_time) * float(
                self.psh_system.parameter['GenMax']) / (float(self.psh_system.parameter['GenEfficiency']) + beta))
            if left_cod or right_cod:
                # if left_value < 0 and right_value > 0:
                point_y = 0  # self.calculate_new_soc(value)
                check = 1
            else:
                # point_y = 0
                point_y = -1000000  # self.calculate_pts(value)
                check = 0
            # FIND the left and right point of using cal_new_soc
            self.second_curve_profit.append(point_y)
            self.check_soc_curve.append(check)

        #抽取需要做multipross的部分在这里计算
        time_1 = time.time()
        initial_soc_list = []
        for i in range(len(self.check_soc_curve)):
            if self.check_soc_curve[i] == 1:
                initial_soc_list.append(self.old_curve.point_X[i])

        MultiRL = MulRLSetUp()
        MultiRL.alpha = self.alpha # 0.2
        MultiRL.date = self.date
        MultiRL.LAC_last_windows = self.LAC_last_windows  # 1#0
        MultiRL.probabilistic = self.probabilistic  # 0#1
        MultiRL.RT_DA = self.RT_DA #1
        MultiRL.curr_time = self.curr_time
        MultiRL.curr_scenario = self.curr_scenario
        MultiRL.current_stage = self.current_stage #'training_500'
        MultiRL.time_period = self.time_period  # 'training_500'
        #initial_soc_list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
        print(initial_soc_list)
        MultiRL.CalOpt(initial_soc_list)

        #把这个放回到profit里面
        k=0
        for i in range(len(self.check_soc_curve)):
            if self.check_soc_curve[i] == 1:
                self.second_curve_profit[i] = MultiRL.optimal_profit_list[k]
                k += 1
        time_2 = time.time()
        print('the time is', time_2 - time_1)





        #边界点的处理，find the boundary point
        self.left = 0
        self.right = len(self.check_soc_curve) - 1

        for item in range(len(self.check_soc_curve)):
            if self.check_soc_curve[0] == 1:
                self.left = 0
            elif item != len(self.check_soc_curve)-1 and (self.check_soc_curve[item] == 0 and self.check_soc_curve[item + 1] == 1):
                self.left = item + 1
            elif item != len(self.check_soc_curve)-1 and (self.check_soc_curve[item] == 1 and self.check_soc_curve[item + 1] == 0):
                self.right = item
            elif item == len(self.check_soc_curve)-1 and self.check_soc_curve[item] == 1:
                self.right = item



        #get new curve_slope
        self.second_curve_slope = [self.old_curve.intial_slope_set]
        for index in range(1, len(self.second_curve_soc)):
            temp_slop = (self.second_curve_profit[index] - self.second_curve_profit[index - 1])/self.curve.steps
            self.second_curve_slope.append(temp_slop)
            #change the first back
        #self.second_curve_slope[0] = self.second_curve_slope.intial_slope_set

        #make sure it is convex
        #注意一边是保持边界的所有是非常大或者非常小,同时要保持slope下降
        for i in range(len(self.second_curve_slope)):
            if i < self.left + 1:
                self.second_curve_slope[i] = 10000 #self.second_curve_slope[self.left + 1]
                #self.old_curve.point_Y[i] = 10000
            elif i == self.left:
                self.second_curve_slope[i] == self.second_curve_slope[self.left + 1]
            elif i > self.right:
                self.second_curve_slope[i] = -10000 #self.second_curve_slope[self.right]
                #self.old_curve.point_Y[i] = 10000
            #保持slope下降


        for i in range(len(self.second_curve_slope)):
            if i == self.left + 1:
                if self.second_curve_slope[i] > self.second_curve_slope[self.left]:
                    self.second_curve_slope[i] = self.second_curve_slope[self.left]
            elif self.left + 1 < i < self.right:
                # if (self.second_curve_slope[i-1] > self.second_curve_slope[i]) and (self.second_curve_slope[i] > self.second_curve_slope[i+1]):
                #     self.second_curve_slope[i] = self.second_curve_slope[i]
                # elif (self.second_curve_slope[i-1] > self.second_curve_slope[i]) and (self.second_curve_slope[i] < self.second_curve_slope[i+1]):
                #     self.second_curve_slope[i] = self.second_curve_slope[i-1]/2 + self.second_curve_slope[i+1] /2
                if self.second_curve_slope[i-1] >= self.second_curve_slope[i+1]:
                    if self.second_curve_slope[i] > self.second_curve_slope[i-1]:
                        self.second_curve_slope[i] = (self.second_curve_slope[i-1]+self.second_curve_slope[i+1])/2
                    elif self.second_curve_slope[i] < self.second_curve_slope[i-1]:
                        self.second_curve_slope[i] = (self.second_curve_slope[i-1]+self.second_curve_slope[i+1])/2
                elif self.second_curve_slope[i-1] < self.second_curve_slope[i+1]:
                    if self.second_curve_slope[i] > self.second_curve_slope[i-1]:
                        self.second_curve_slope[i] = self.second_curve_slope[i-1]
            elif i == self.right:
                if self.second_curve_slope[i] > self.second_curve_slope[i-1]:
                    self.second_curve_slope[i] = self.second_curve_slope[i-1]


        print(self.second_curve_slope)

            # _cur = len(self.second_curve_slope) - i - 1
            # if self.check_soc_curve[i] == 0:
            #     if _cur != 0 and self.second_curve_slope[_cur] > self.second_curve_slope[_cur-1] and self.second_curve_slope[_cur] < self.old_curve.intial_slope_set:
            #         self.second_curve_slope[_cur - 1] = self.second_curve_slope[_cur]
            #     elif _cur != 0 and self.second_curve_slope[_cur] > self.second_curve_slope[_cur-1] and self.second_curve_slope[_cur] > self.old_curve.intial_slope_set:
            #         self.second_curve_slope[_cur] = self.old_curve.intial_slope_set
            #         self.second_curve_slope[_cur - 1] = self.old_curve.intial_slope_set





    def get_new_curve_step_2_curve_comb(self):
    #new curve combine with the old_slope
        self.new_curve_slope = []
        for i in range(len(self.second_curve_soc)):
            _temp = (1 - self.alpha)*self.old_curve.point_Y[i] + self.alpha*self.second_curve_slope[i]
            self.new_curve_slope.append(_temp) #this is the new slope we need
        print(self.new_curve_slope)

        # make sure it is convex
        # 注意一边是保持边界的所有是非常大或者非常小,同时要保持slope下降
        # for i in range(len(self.new_curve_slope)):
        #     # 保持slope下降
        #     if i == self.left + 1:
        #         if self.new_curve_slope[i] > self.new_curve_slope[self.left]:
        #             self.new_curve_slope[i] = self.new_curve_slope[self.left]
        #     elif self.left + 1 < i <= self.right:
        #         if self.new_curve_slope[i] > self.new_curve_slope[i - 1]:
        #             self.new_curve_slope[i] = self.new_curve_slope[i - 1]

    def get_new_curve_step_3_two_pts(self):
        #need find another point #be careful boundary case
        # 注意你的点是以后面为标准
        self.second_point_soc_sum = self.optimal_soc_sum + self.curve.steps
        self.second_point_soc_sum_0 = self.second_point_soc_sum - 1

        self.first_point_soc_sum = self.optimal_soc_sum
        self.first_point_soc_sum_0 = self.optimal_soc_sum - 1

        if self.first_point_soc_sum_0 >= self.curve.lo_bd and self.second_point_soc_sum <= self.curve.up_bd:
            self.second_point_soc_sum_profit = self.calculate_new_soc(self.second_point_soc_sum)
            self.second_point_soc_sum_profit_0 = self.calculate_new_soc(self.second_point_soc_sum_0)
            self.first_point_soc_sum_profit = self.calculate_new_soc(self.first_point_soc_sum)
            self.first_point_soc_sum_profit_0 = self.calculate_new_soc(self.first_point_soc_sum_0)
            self.update_point_1_x = self.first_point_soc_sum
            self.update_point_1_y = self.first_point_soc_sum_profit - self.first_point_soc_sum_profit_0
            self.update_point_2_x = self.second_point_soc_sum
            self.update_point_2_y = self.second_point_soc_sum_profit - self.second_point_soc_sum_profit_0
        elif self.first_point_soc_sum_0 < self.curve.lo_bd and self.second_point_soc_sum < self.curve.up_bd:
            #只有左边出界了
            self.second_point_soc_sum_profit = self.calculate_new_soc(self.second_point_soc_sum)
            self.second_point_soc_sum_profit_0 = self.calculate_new_soc(self.second_point_soc_sum_0)
            # self.first_point_soc_sum_profit = self.calculate_new_soc(self.first_point_soc_sum)
            # self.first_point_soc_sum_profit_0 = self.calculate_new_soc(self.first_point_soc_sum + 1)

            self.update_point_2_x = self.second_point_soc_sum
            self.update_point_2_y = self.second_point_soc_sum_profit - self.second_point_soc_sum_profit_0
            # 这个地方是不是有小问题？前后也不一样呀？
            self.update_point_1_x = self.update_point_2_x   #self.first_point_soc_sum
            self.update_point_1_y = self.update_point_2_y   #self.first_point_soc_sum_profit_0 - self.first_point_soc_sum_profit
        elif self.first_point_soc_sum_0 >= self.curve.lo_bd and self.second_point_soc_sum > self.curve.up_bd:
            # self.second_point_soc_sum_profit = self.calculate_new_soc(self.second_point_soc_sum)
            # self.second_point_soc_sum_profit_0 = self.calculate_new_soc(self.second_point_soc_sum_0)
            self.first_point_soc_sum_profit = self.calculate_new_soc(self.first_point_soc_sum)
            self.first_point_soc_sum_profit_0 = self.calculate_new_soc(self.first_point_soc_sum_0)
            self.update_point_1_x = self.first_point_soc_sum
            self.update_point_1_y = self.first_point_soc_sum_profit - self.first_point_soc_sum_profit_0
            self.update_point_2_x = self.update_point_1_x
            self.update_point_2_y = self.update_point_1_y

        #用convex
        #alpha = 0.8
        #self.update_point_1_y = (1 - alpha) * (self.calculate_pts(self.update_point_1_x)- self.calculate_pts(self.update_point_1_x-1)) + alpha * self.update_point_1_y
        #self.update_point_2_y = (1 - alpha) * (self.calculate_pts(self.update_point_2_x)- self.calculate_pts(self.update_point_2_x-1)) + alpha * self.update_point_2_y

        self.update_point_1 = [self.update_point_1_x, self.update_point_1_y]
        self.update_point_2 = [self.update_point_2_x, self.update_point_2_y]

        #four points: previous_point, optimal_soc, second_point,
        # if self.optimal_soc_sum + 1 > self.curve.up_bd:
        #     self.second_point_soc_sum = self.optimal_soc_sum - 1  #self.curve.steps
        #     self.second_point_profit = self.calculate_new_soc(self.second_point_soc_sum)
        # else:
        #     self.second_point_soc_sum = self.optimal_soc_sum + 1  #self.curve.steps
        #     self.second_point_profit = self.calculate_new_soc(self.second_point_soc_sum)
        #
        # # get previous point profit
        # if self.optimal_soc_sum - 1 < self.curve.lo_bd:
        #     self.previous_point_soc_sum = self.optimal_soc_sum + 1 #self.curve.steps
        #     self.previous_point_profit = self.calculate_new_soc(self.previous_point_soc_sum)
        # else:
        #     self.previous_point_soc_sum = self.optimal_soc_sum - 1 #self.curve.steps
        #     self.previous_point_profit = self.calculate_new_soc(self.previous_point_soc_sum)
        # # shall we get the optimal at previous???
        # self.pre_scen_optimal_profit = self.calculate_new_soc(self.optimal_soc_sum)
        #
        # #calcuate self.update_point_1/2(point_x, point_curve)
        # if self.optimal_soc_sum + 1 > self.curve.up_bd:
        #     # self.optimal_profit and self.optimal_soc_sum
        #     self.update_point_1_x = self.optimal_soc_sum
        #     self.update_point_1_y = (self.pre_scen_optimal_profit - self.previous_point_profit) #self.curve.steps
        #     #
        #     self.update_point_2_x = self.optimal_soc_sum
        #     self.update_point_2_y = (self.pre_scen_optimal_profit - self.previous_point_profit) #self.curve.steps
        #
        # elif self.optimal_soc_sum - 1 < self.curve.lo_bd:
        #     #self.optimal_profit and self.optimal_soc_sum
        #     self.update_point_1_x = self.optimal_soc_sum
        #     self.update_point_1_y = (self.second_point_profit - self.pre_scen_optimal_profit) #self.curve.steps
        #     ##这里写错了，到底是update前面的点，还是这个点？
        #     self.update_point_2_x = self.optimal_soc_sum
        #     self.update_point_2_y = (self.second_point_profit - self.pre_scen_optimal_profit) #self.curve.steps
        # else:
        #     self.update_point_1_x = self.optimal_soc_sum
        #     self.update_point_1_y = (self.pre_scen_optimal_profit - self.previous_point_profit) #self.curve.steps
        #     self.update_point_2_x = self.second_point_soc_sum
        #     self.update_point_2_y = (self.second_point_profit - self.pre_scen_optimal_profit) #self.curve.steps
        # self.update_point_1 = [self.update_point_1_x, self.update_point_1_y]
        # self.update_point_2 = [self.update_point_2_x, self.update_point_2_y]


    # def get_new_curve_step_5_curve_comb(self):
    # #new curve combine with the old_slope
    #     self.new_curve_slope = []
    #     for i in range(len(self.second_curve_soc)):
    #         _temp = (1 - self.alpha)*self.old_curve.point_Y[i] + self.alpha*self.second_curve_slope[i]
    #         self.new_curve_slope.append(_temp) #this is the new slope we need
    #     print(self.new_curve_slope)



    def output_curve(self):
    #output the curve
        scenario = self.curr_scenario
        filename = self.e_system.e_start_folder + '/Curve_' + 'time_' + str(self.curr_model_para.LAC_bhour) + '_scenario_' +  str(scenario) + '.csv'
        df = pd.DataFrame(self.curve.segments, columns =['soc_segment','slope'])
        df.to_csv(filename, index=False, header=True)


    def output_curve_sum(self):
        #input the original
        curr_time = self.curr_model_para.LAC_bhour
        scenario = self.curr_model_para.scenario
#output_curve_sum这里有问题
        if scenario == 1:
            filename = self.e_system.e_start_folder + '/Curve_' + 'time_' + str(curr_time) + '_scenario_' + str(scenario) + '.csv'
            df = pd.read_csv(filename)
        else:
            filename = self.e_system.e_start_folder + '/Curve_total_' + 'time_' + str(self.curr_model_para.LAC_bhour) + '.csv'
            df = pd.read_csv(filename)
        #output the current


        #df_cur = pd.DataFrame(self.curve.segments, columns=['soc_segment', 'slope_time_' + str(curr_time) + str(scenario)])
        df_cur = pd.DataFrame(self.curve.point_Y, columns=['slope_time_' + str(curr_time) + str(scenario)])
        df = pd.concat([df, df_cur], axis = 1)

        filename = self.e_system.e_start_folder + '/Curve_total_' + 'time_' + str(self.curr_model_para.LAC_bhour)+'.csv'
        df.to_csv(filename, index=False, header=True)


    def calculate_pts(self, point_X):
        #put the soc_sum in, we get the profit
        point_x_soc = self.x_to_soc(point_X)
        point_profit = []
        for s in range(self.lmp.Nlmp_s):
            p_s = self.lmp.lmp_quantiles[s]
            for j in self.psh_system.parameter['PSHName']:
                point_profit.append((self.optimal_psh_gen_sum - self.optimal_psh_pump_sum) * self.lmp.lmp_scenarios[s][0] * p_s)
        # for j in self.psh_system.parameter['PSHName']:
        #     point_profit.append((self.psh_gen[j] - self.psh_pump[j]) * self.lmp.lmp_scenarios[0][0])

        #self.curr_cost = sum(point_profit)
        for k in self.e_system.parameter['EName']:
            for i in range(self.curve.numbers):
                bench_num = i
                point_profit.append(self.curve.point_Y[bench_num + 1] * point_x_soc[bench_num])
        point_profit_sum = sum(point_profit)
        return point_profit_sum




    def x_to_soc(self, point_X):
        #change soc_sum to soc_1 + soc_2 + soc_3
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





train = RL_Kernel()
#test.calculate_old_curve()
date_list =['March 07 2019', 'April 01 2019', 'April 15 2019', 'April 22 2019']
#alpha = [0, 0.2, 0.5, 0.8, 1]
#date_list =['April 22 2019']
alpha = [0.2]
#test.end = 100
train.LAC_last_windows = 0  # 0#1 #必须是1才可以是DA的price
train.probabilistic = 1  # 1#0
train.RT_DA = 0  # 1#0
#test.main_function()
for i in range(len(date_list)):
    for j in range(len(alpha)):
        train.alpha = alpha[j]
        train.date = date_list[i]
        train.main_function()





from ModelSetUp import *


class Prediction():
    def __init__(self):
        #self.reward = None
        #self.value = None
        #self.action = None
        self.alpha = None #0, 0.2 , 0.5, 0.8, 1
        self.date = None#'March 07 2019'
        self.LAC_last_windows = None #0#1 #必须是1才可以是DA的price
        self.probabilistic = None #1#0
        self.RT_DA = None #1#0
        self.curr_time = None
        self.curr_scenario = None
        self.current_stage ='sample' #'training_500'
            # #用sample就调整位置了，需不需要专门一个来记录位置的？
        #如果我们要用repetitive DA， 我们需要LAC_last_windows = 0， probabilitsit = 1, DA = 0?
        self.time_period = 23 #24-1

    def main_function(self):
        self.Curr_Scenario_Cost_Total = []
        self.start = 1
        self.end = 5
        for curr_scenario in range(self.start, self.end):
            self.PSH_Results = []
            self.SOC_Results = []
            self.curr_scenario_cost_total = 0
            for i in range(self.time_period):
                self.curr_time = i
                self.curr_scenario = curr_scenario
                self.calculate_optimal_soc()
                #self.get_final_curve_main()
                self.output_psh_soc()
            self.output_psh_soc_main()
        self.output_curr_cost()



    def calculate_optimal_soc(self):
        self.curr_model_para = CurrModelPara(self.LAC_last_windows, self.probabilistic, self.RT_DA, self.date, self.curr_time, self.curr_scenario, self.current_stage, self.time_period)
        # LAC_last_windows,  probabilistic, RT_DA, date, LAC_bhour, scenario

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
        self.lmp.predict_set_up_parameter()
        #print(self.lmp.date)
        #print('lmp_quantiles=', self.lmp.lmp_quantiles)
        #print('lmp_scenarios=', self.lmp.lmp_scenarios)
        #print('lmp_Nlmp_s=', self.lmp.Nlmp_s)

        print('################################## curve set up ##################################')
        self.old_curve = Curve(100, 0, 3000, self.time_period)


        ####不同的开始，不同的curve
        # if self.curr_scenario == 1 and self.curr_time == 0:
        #     self.old_curve.output_initial_curve()
        # #
        # if self.LAC_last_windows == 0 and self.probabilistic == 1 and self.RT_DA == 0 and self.curr_scenario == 1:
        #     last_scenario = 10000
        #     self.old_curve.input_tuned_initial_curve(last_scenario)
        # self.old_curve.input_curve(self.curr_time, self.curr_scenario - 1)


        #choose the scenario you need
        prediction_scenario = 2398
        self.old_curve.input_prediction_curve(prediction_scenario, self.curr_time)
        print(self.old_curve.segments)
        print(self.old_curve.point_Y)

        print('################################## ADP training model set up ##################################')
        model_1 = Model('DAMarket')
        self.curr_model = RLSetUp(self.psh_system, self.e_system, self.lmp, self.old_curve, self.curr_model_para, model_1)
        self.curr_model.optimization_model()
        self.optimal_soc_sum = self.curr_model.optimal_soc_sum
        self.optimal_psh_gen_sum = self.curr_model.optimal_psh_gen_sum
        self.optimal_psh_pump_sum = self.curr_model.optimal_psh_pump_sum

        #print(self.curr_model.optimal_soc_sum)

    def output_psh_soc_main(self):
        # add the last one

        #filename = './Output_Curve' + '/PSH_Profitmax_Rolling_Results_' + str(self.curr_scenario) + '_' + self.date + '.csv'
        if self.SOC_Results[-1] - self.e_system.parameter['EEnd'][0] > 0.1:
            self.PSH_Results.append(
                (self.SOC_Results[-1] - self.e_system.parameter['EEnd'][0]) *
                self.psh_system.parameter['GenEfficiency'][0])
        else:
            self.PSH_Results.append(
                (self.SOC_Results[-1] - self.e_system.parameter['EEnd'][0]) /
                self.psh_system.parameter['PumpEfficiency'][0])

        self.SOC_Results.append(self.e_system.parameter['EEnd'][0])

        self.df = pd.DataFrame(
            {'SOC_Results_' + str(self.curr_scenario): self.SOC_Results,
             'PSH_Results_' + str(self.curr_scenario): self.PSH_Results})
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

        ##output curr cost
        self.curr_scenario_cost_total += self.curr_model.curr_cost


    def output_curr_cost(self):
        # output the psh and soc
        filename = './Output_Curve' + '/PSH_Profitmax_Prediction_Results_' + 'total' + '_' + self.date +'_alpha_' + str(int(self.alpha*10)) +'.csv'
        self.df_total.to_csv(filename)

        # output curr_cost
        filename = './Output_Curve' + '/PSH_Profitmax_Prediction_Current_Cost_Results_' + str(
            self.curr_scenario) + '_' + self.date +'_alpha_' + str(int(self.alpha*10)) + '.csv'
        self.df = pd.DataFrame({'Curr_Scenario_Cost_Total': self.Curr_Scenario_Cost_Total})
        self.df.to_csv(filename)


test = Prediction()
#test.calculate_old_curve()
#date_list =['March 07 2019', 'April 01 2019', 'April 15 2019', 'April 22 2019']
#alpha = [0, 0.2, 0.5, 0.8, 1]
date_list =['March 07 2019']
alpha = [0.2]
#test.end = 100
test.LAC_last_windows = 0  # 0#1 #必须是1才可以是DA的price
test.probabilistic = 0  # 1#0
test.RT_DA = 0  # 1#0
#010 #100
#test.main_function()
for i in range(len(date_list)):
    for j in range(len(alpha)):
        test.alpha = alpha[j]
        test.date = date_list[i]
        test.main_function()
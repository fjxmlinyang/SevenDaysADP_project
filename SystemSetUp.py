import pandas as pd
from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import numpy as np
from Curve import *
from CurrModelPara import *





# class Folder_Info():
#     def __init__(self, Input_folder_parent, Output_folder, curr_model):
#         self.curr_model = curr_model
#         self.Input_folder_parent = Input_folder_parent
#         self.Output_folder = self.Output_folder
#         self.date = self.curr_model.date
#         self.Input_folder = self.Input_folder_parent + '/' +self.date
#         self.filename = None


class System():
    def __init__(self, curr_model):
        self.curr_model = curr_model
        self.Input_folder_parent = None
        self.filename = None
        self.Input_folder = None
        self.Output_folder = None
        self.parameter = {}
        #this is for easy
        if curr_model.current_stage == 'training_50':
            self.Input_all_total = './Input_Curve'
        if curr_model.current_stage == 'training_500':
            self.Input_all_total = './Input_bootstrap'
        if curr_model.current_stage == 'test':
            self.Input_all_total = './Input_test'
        if curr_model.current_stage == 'sample':
            self.Input_all_total = './Input_prediction'

    def input_parameter(self, paranameter_name, in_model_name):
        Data = pd.read_csv(self.filename)
        df = pd.DataFrame(Data)
        # ret = list(df[paranameter_name])
        ret = df[paranameter_name]
        self.parameter[in_model_name] = ret  # [0]


class PshSystem(System):

    def set_up_parameter(self):
##这个是给标量 #how to input one by one?
        self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
        self.Input_folder = self.Input_folder_parent +'/'+ self.curr_model.date
        self.filename = self.Input_folder +'/PSH.csv'
        self.input_parameter('GenMin', 'GenMin')
        self.input_parameter('GenMax', 'GenMax')
        self.input_parameter('PumpMin', 'PumpMin')
        self.input_parameter('PumpMax', 'PumpMax')
        self.input_parameter('Cost', 'Cost')
        self.input_parameter('Efficiency', 'GenEfficiency')
        self.input_parameter('Efficiency', 'PumpEfficiency')
        self.input_parameter('Name', 'PSHName')       

        self.Input_folder = None
        self.filename = None




class ESystem(System):

    def set_up_parameter(self):
##这个是给标量 #how to input one by one?
        self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
        self.Input_folder = self.Input_folder_parent +'/'+ self.curr_model.date 
        self.filename = self.Input_folder +  '/Reservoir.csv'
        self.input_parameter('Min', 'EMin')
        self.input_parameter('Max', 'EMax')
        self.input_parameter('Name', 'EName') 
        self.input_parameter('End', 'EEnd')

        self.Output_folder='./Output_Curve'
        #here are for rolling model
        #here we can set the benchmark?
        if self.curr_model.LAC_bhour == 0:
            self.input_parameter('Start', 'EStart')
            self.e_start_folder = self.Output_folder
        elif self.curr_model.LAC_last_windows:
            self.filename = self.Output_folder + '/LAC_Solution_System_SOC_' + str(self.curr_model.LAC_bhour - 1) + '.csv'
            self.input_parameter('SOC', 'EStart')
            self.e_start_folder = self.Output_folder
        else:
            self.filename = self.Output_folder + '/LAC_Solution_System_SOC_' + str(self.curr_model.LAC_bhour - 1) + '.csv'
            self.input_parameter('SOC', 'EStart')
            self.e_start_folder = self.Output_folder
        self.Input_folder = None
        self.filename = None
        self.Output_folder = None



class LMP(System):

    def set_up_parameter(self):
        self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
        self.Input_folder = self.Input_folder_parent+'/'+ self.curr_model.date
        if self.curr_model.LAC_last_windows:
            # filename = Input_folder + '\LMP_Hindsight' + '.csv'
            #self.filename = self.Input_folder + '/prd_dataframe_wlen_24_'+ self.curr_model.date + '.csv'
            self.filename = self.Input_folder + '/prd_dataframe_wlen_' + str(
                self.curr_model.time_period + 1  - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '.csv'
        else:
            # filename = Input_folder+'\LMP_Scenarios_' + 'T' + str(LAC_bhour) +'_DA'+ '.csv'
            if self.curr_model.probabilistic and self.Input_all_total == './Input_bootstrap':
                self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(24-self.curr_model.LAC_bhour) + '_'+ self.curr_model.date+'_550' + '.csv'
            elif self.curr_model.probabilistic and self.Input_all_total == './Input_test':
                self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(24-self.curr_model.LAC_bhour) + '_'+ self.curr_model.date+'_550' + '.csv'
            elif self.curr_model.probabilistic and self.Input_all_total == './Input_sample':
                self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(self.curr_model.time_period + 1 - self.curr_model.LAC_bhour) + '_' + self.curr_model.date + '_50' + '.csv'
            elif self.curr_model.probabilistic and self.Input_all_total == './Input_Curve':
                self.filename = self.Input_folder + '/DA_lmp_Scenarios_wlen_' + str(24-self.curr_model.LAC_bhour) + '_'+ self.curr_model.date+'_50' + '.csv'
            else:
                self.filename = self.Input_folder + '/prd_dataframe_wlen_'+ str(24-self.curr_model.LAC_bhour)+'_'+ self.curr_model.date + '.csv'
        
        
        Data = pd.read_csv(self.filename)
        df = pd.DataFrame(Data)
        Column_name = list(Data.columns)
        self.lmp_quantiles = []
        self.lmp_scenarios = []
        #DA_lmp=[]???
        if self.curr_model.LAC_last_windows:
            self.Nlmp_s = 1
            # probability of each scenario is evenly distributed
            self.lmp_quantiles.append(1.0 / self.Nlmp_s)
            if self.curr_model.RT_DA == 1:
                self.lmp_scenarios.append(list(df['RT_LMP']))
            else:
                self.lmp_scenarios.append(list(df['DA_LMP']))
        else:
            if self.curr_model.probabilistic:
                self.Nlmp_s = 1
                self.lmp_quantiles.append(1.0 / self.Nlmp_s)
                read_curr = (self.curr_model.scenario) % 50 - 1
                self.lmp_scenarios.append(list(df[Column_name[read_curr]]))

    #            self.Nlmp_s=len(Column_name)
    #             for i in range(self.Nlmp_s):
    #                 # probability of each scenario is evenly distributed
    #                 self.lmp_quantiles.append(1.0 / self.Nlmp_s)
    # ##only change here!!!
    #                 #read_curr = self.curr_model.scenario
    #                 read_curr = (self.curr_model.scenario) % 50 -1
    #                 self.lmp_scenarios.append(list(df[Column_name[read_curr]]))

            else:
                # for deterministic forecast, there is a single scenario
                self.Nlmp_s = 1
                self.lmp_quantiles.append(1.0 / self.Nlmp_s)
                # deterministic forecast is the single point prediction
                self.lmp_scenarios.append(list(df['prd']))
            #所以要用scneario,我们需要LAC_last_windows = 0, probabilistic = 1, DA = 0
            #如果我们要用repetitive DA， 我们需要LAC_last_windows = 0， probabilitsit = 1, DA = 0?


        self.Input_folder = None
        self.filename = None
        self.Output_folder = None

    def predict_set_up_parameter(self):

        self.Input_folder_parent = self.Input_all_total + '/PSH-Rolling Window'
        self.Input_folder = self.Input_folder_parent + '/' + self.curr_model.date

        self.filename = self.Input_folder + '/Prediction_lmp_Scenarios_wlen_' + str(
            self.curr_model.time_period + 1 - self.curr_model.LAC_bhour) +'_50' + '.csv'
        Data = pd.read_csv(self.filename)
        df = pd.DataFrame(Data)
        Column_name = list(Data.columns)
        self.lmp_quantiles = []
        self.lmp_scenarios = []
        # DA_lmp=[]???
        self.Nlmp_s = 1
        self.lmp_quantiles.append(1.0 / self.Nlmp_s)
        read_curr = (self.curr_model.scenario - 1)
        self.lmp_scenarios.append(list(df[Column_name[read_curr]]))

        self.Input_folder = None
        self.filename = None
        self.Output_folder = None





#
# psh_folder_info = Folder_Info(Input_folder_parent, Output_folder, curr_model)
# Curr_Model(LAC_last_windows是否是最后,  probabilistic是否是随机模型, RT_DA是否是RT, date, LAC_bhour开始时间)
#

# psh_model = CurrModelPara(1, 0 , 1,'March 07 2019', 1,1)   ##first is for last window second are stochastic, third is for the date, last is for hour
# psh_system = PshSystem(psh_model)
# psh_system.set_up_parameter()
# psh_system.parameter['EStart']= 1000000
# print(psh_system.parameter)
#
# e_model = CurrModelPara(1, 0 ,1, 'March 07 2019', 1,1)   ##first is for last window second are stochastic, third is for the date, last is for hour
# e_system = ESystem(e_model)
# e_system.set_up_parameter()
# print(e_system.parameter)
#
#
# lmp_model = CurrModelPara(1, 0 ,1, 'March 07 2019', 1, 1)
# print(lmp_model.date)  ##first is for last window second are stochastic, third is for the date, last is for hour
# lmp = LMP(lmp_model)
# lmp.set_up_parameter()
# print(lmp.lmp_quantiles)









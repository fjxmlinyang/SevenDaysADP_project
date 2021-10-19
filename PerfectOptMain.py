import pandas as pd
from PerfectOpt import *
#from LAC_PSH_profit_max import PSH_profitmax_plot

###################################### Set input and output folder and parameters ##############################################

Input_folder = './DataMain'

time_total = 24 *7 - 1

Output_folder = './Output_Perfect'
# set the length of the rolling window
LAC_window = 1
# indicate if the current window is the last window, default as 0
LAC_last_windows = 0
# 0:apply the deterministic forecast, 1:apply the probabilistic forecast
probabilistic = 0
# read time periods

#day
date = 'October 1521'

time_periods = []
filename = Input_folder + '/DECO_prd_dataframe_Perfect_October 1521 2019.csv'
Data = pd.read_csv(filename)
df = pd.DataFrame(Data)

#the dataframe column can be multiple, for example, RT_LMP, scenario_1, ......
LMP_Scenario = df['RT_LMP']
for i in range(len(LMP_Scenario)-1):
    time_periods.append('T' + str(i))

dict = {}
PSH_Results = []
SOC_Results = []
Price_Results = []
Curr_Profit = []
Case_Total = 3


for i in range(Case_Total):
    scenario = i + 1
    Total_hour = len(time_periods)
    LAC_last_windows=1

    RT_DA=0
    PSH_Profitmax = Perfect_Opt(Total_hour, Input_folder, Output_folder, date, RT_DA, probabilistic, time_total, scenario)
    SOC_Results = PSH_Profitmax[0]
    PSH_Results = PSH_Profitmax[1]
    Price_Results = PSH_Profitmax[2]
    filename = Input_folder + '/PSH.csv'
    Data = pd.read_csv(filename)
    df = pd.DataFrame(Data)
    PSHefficiency = list(df['Efficiency'])

    filename = Input_folder + '/Reservoir.csv'
    Data = pd.read_csv(filename)
    df = pd.DataFrame(Data)
    Eend = float(df['End'])


    dict['V' + str(scenario) + '_Price'] = Price_Results
    dict['V' + str(scenario) + '_SOC'] = SOC_Results
    dict['V' + str(scenario) + '_PSH'] = PSH_Results
    curr_profit = 0
    for i in range(len(PSH_Results)):
        curr_profit += PSH_Results[i] * Price_Results[i]
    Curr_Profit.append(curr_profit)

    # write results
filename = Output_folder + '/PSH_Profitmax_Rolling_Results_'+ date +'.csv'
df = pd.DataFrame(dict)
df.to_csv(filename)

filename = Output_folder + '/Curr_Profitmax_Rolling_Results_'+ date +'.csv'
df = pd.DataFrame(Curr_Profit)
df.to_csv(filename)



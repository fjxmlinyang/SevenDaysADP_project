import pandas as pd
from PerfectOpt import *
#from LAC_PSH_profit_max import PSH_profitmax_plot

###################################### Set input and output folder and parameters ##############################################

Input_folder_parent='./Input_Curve/PSH-Rolling Window'
## Pick a date
#date='March 07 2019'
date='April 01 2019'
#date='April 15 2019'

time_total = 24
Input_folder=Input_folder_parent+'/'+date
Output_folder='./Output'
# set the length of the rolling window
LAC_window=1
# indicate if the current window is the last window, default as 0
LAC_last_windows=0
# 0:apply the deterministic forecast, 1:apply the probabilistic forecast
probabilistic=0
# read time periods
time_periods=[]
filename = Input_folder + '/prd_dataframe_wlen_'+str(time_total)+'_'+date+'.csv'
Data = pd.read_csv(filename)
df = pd.DataFrame(Data)
LMP_Scenario= df['RT_LMP']
for i in range(len(LMP_Scenario)-1):
    time_periods.append('T' + str(i))

###################################### Rolling window starts ##############################################

# for scen in range(50):
#     scenario = scen + 1
#     PSH_Results = []
#     SOC_Results = []
#     Price_Results = []
#     #only one?
#     for i in time_periods:
#         start_hour=time_periods.index(i)
#         LAC_bhour = start_hour
#         if start_hour+LAC_window <= len(time_periods):
#             # 1:RT price data, 0: DA price data
#             RT_DA=0
#             PSH_Profitmax = Perfect_Opt(LAC_bhour, LAC_last_windows, Input_folder, Output_folder, date, RT_DA, probabilistic, time_total, scenario)
#             PSH_Results.append(PSH_Profitmax[1][0])
#             SOC_Results.append(PSH_Profitmax[0][0])
#             Price_Results.append(PSH_Profitmax[2][0])
#
#     filename = Input_folder + '/PSH.csv'
#     Data = pd.read_csv(filename)
#     df = pd.DataFrame(Data)
#     PSHefficiency = list(df['Efficiency'])
#
#     filename = Input_folder + '/Reservoir.csv'
#     Data = pd.read_csv(filename)
#     df = pd.DataFrame(Data)
#     Eend = float(df['End'])
#     PSH_Results.append((SOC_Results[-1] - Eend) / PSHefficiency[0])
#     SOC_Results.append(Eend)
#     dict['V' + str(scenario) + '_Price'] = Price_Results[0]
#     dict['V' + str(scenario) + '_SOC'] = SOC_Results
#     dict['V' + str(scenario) + '_PSH'] = PSH_Results

######################################Find the perfect result ##############################################
dict = {}
PSH_Results = []
SOC_Results = []
Price_Results = []
Curr_Profit = []
for i in range(50):
    scenario = i + 1
    LAC_bhour=len(time_periods)
    LAC_last_windows=1
    # Real-time benchmark using the after the fact RT price
    # RT_DA=1
    # PSH_Profitmax = LAC_PSH_Profitmax(LAC_bhour, LAC_last_windows, Input_folder, Output_folder,date,RT_DA,probabilistic, time_total, scenario)
    # After_fact_SOC=PSH_Profitmax[0]
    # After_fact_PSH =PSH_Profitmax[1]
    # DA benchmark using the DA price
    RT_DA=0
    PSH_Profitmax = Perfect_Opt(LAC_bhour, LAC_last_windows, Input_folder, Output_folder,date,RT_DA,probabilistic, time_total, scenario)
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


    dict['V' + str(scenario) + '_Price'] = Price_Results[0]
    dict['V' + str(scenario) + '_SOC'] = SOC_Results
    dict['V' + str(scenario) + '_PSH'] = PSH_Results
    curr_profit = 0
    for i in range(len(PSH_Results)):
        curr_profit += PSH_Results[i] * Price_Results[0][i]
    Curr_Profit.append(curr_profit)

    # write results
filename = Output_folder + '/PSH_Profitmax_Rolling_Results_'+ date +'.csv'
df = pd.DataFrame(dict)
df.to_csv(filename)

filename = Output_folder + '/Curr_Profitmax_Rolling_Results_'+ date +'.csv'
df = pd.DataFrame(Curr_Profit)
df.to_csv(filename)





# df =pd.DataFrame({'After_fact_SOC':After_fact_SOC, 'SOC_Results':SOC_Results,'DA_SOC':DA_SOC,
#                   'After_fact_PSH':After_fact_PSH, 'PSH_Results':PSH_Results, 'DA_PSH':DA_PSH})
# df.to_csv(filename)
#
# ###################################### Plot Section ##############################################RT_DA=1
# # benchmark with RT after-the-fact results
# RT_DA=1
# PSH_profitmax_plot(Input_folder,Output_folder,date,RT_DA,probabilistic)
# # benchmark with DA results
# RT_DA=0
# PSH_profitmax_plot(Input_folder,Output_folder,date,RT_DA,probabilistic)
#



#section 1只算一个得
#section 2output所有的

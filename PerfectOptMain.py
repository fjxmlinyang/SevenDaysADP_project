import pandas as pd
from PerfectOpt import *

###################################### Set input and output folder and parameters ##############################################

Input_folder = './DataMain'

Output_folder = './Output_Perfect'

#day
date = 'October 1521'



dict = {}
PSH_Results = []
SOC_Results = []
Price_Results = []
Curr_Profit = []
Case_Total = 2 #this is the perfect case you would like to run


for i in range(Case_Total):
    scenario = i + 1
    PSH_Profitmax = Perfect_Opt(Input_folder, scenario)
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



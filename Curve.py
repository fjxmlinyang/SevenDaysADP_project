import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from csv import reader

class Curve(object):
    def __init__(self, numbers, lo_bd, up_bd, time_period):
        self.numbers = numbers
        self.up_bd = up_bd
        self.lo_bd = lo_bd
        self.steps = (up_bd-lo_bd) // numbers
        self.filename_all = './Output_Curve'
        self.seg_initial()
        self.curve_initial()
        self.time_period = time_period
        #self.output_initial_curve()

    
    def seg_initial(self):
        segments = []
        for i in range(self.lo_bd, self.up_bd + self.steps, self.steps):
            curr_step = i // self.steps
            if i == self.lo_bd:
                value = 50
                self.intial_slope_set = value
            else:
                value =  value - 0.02*self.steps #/10
                #value = 50 - curr_step *0.4
            #value = (100 - 2*i // self.steps)
            segments.append([i, value])
        self.segments = segments
    
    def seg_update(self, point_1, point_2):
        point_1_x = point_1[0]
        point_1_y = point_1[1]
        point_2_x = point_2[0]
        point_2_y = point_2[1]
        for i in range(self.numbers + 1):
            curr = self.segments[i]
            curr_x = curr[0]
            curr_y = curr[1]
            if curr_x <= point_1_x and curr_y <= point_1_y:
                self.segments[i][1] = point_1_y
            elif curr_x >= point_2_x and curr_y >= point_2_y:
                self.segments[i][1] = point_2_y
        self.curve_initial()  #需要把point_X and point_Y更新下
        print(self.segments)

    def curve_initial(self):
        df = pd.DataFrame(self.segments, columns=['x','y'])
        self.curve_df = df
        self.point_X = self.curve_df['x'].to_list()
        self.point_Y = self.curve_df['y'].to_list()

    def show_curve(self):
        sns.set_theme(style="darkgrid")   
        sns.lineplot(x='x', y='y', data=self.curve_df)
        plt.show()


    def curve_update(self, new_curve_Y, point_1, point_2):
        for i in range(len(new_curve_Y)):
            value = new_curve_Y[i]
            self.segments[i][1] = value
        self.seg_update(point_1, point_2)

    def input_curve(self, time, scenario):
        _str = str(time)
        #filename = self.filename_all + '/Curve_' + 'time_' + _str + '_scenario_' + str(scenario) + '.csv'
        filename = f'{self.filename_all}/Curve_time_{_str}_scenario_{str(scenario)}.csv'
        df = pd.read_csv(filename)
        self.segments = df.values.tolist()
        self.curve_initial() #!!!别忘了
        print(self.segments)

    def output_initial_curve(self):
        # output the initial curve
        for curr_time in range(self.time_period):
            _str = str(curr_time)
            scenario = 0
            filename = self.filename_all + '/Curve_' + 'time_' + _str + '_scenario_' + str(scenario) + '.csv'
            df = pd.DataFrame(self.segments, columns=['soc_segment', 'slope'])
            df.to_csv(filename, index=False, header=True)

    def input_tuned_initial_curve(self, last_scenario):
        # only works when this project starts
        # read 名字
        for curr_time in range(self.time_period):
            _str = str(curr_time)
            filename = f'{self.filename_all}/Curve_time_{_str}_scenario_{str(last_scenario)}.csv'
            df = pd.read_csv(filename)
            # self.segments = df.values.tolist()
            # self.curve_initial() #!!!别忘了
            scenario = 0
            filename = self.filename_all + '/Curve_' + 'time_' + _str + '_scenario_' + str(scenario) + '.csv'
            df.to_csv(filename, index=False, header=True)

    def input_prediction_curve(self, last_scenario, curr_time):
        #only works when this project starts
        #read 名字
        _str = str(curr_time)
        filename = f'{self.filename_all}/Curve_time_{_str}_scenario_{str(last_scenario)}.csv'
        df = pd.read_csv(filename)
        self.segments = df.values.tolist()
        self.curve_df = df
        self.point_X = self.curve_df['soc_segment'].to_list()
        self.point_Y = self.curve_df['slope'].to_list()
        print(self.point_Y)

# curve_1 = Curve(100, 0, 3000)
#
# print(curve_1.segments)
# #curve_1.input_curve()
# print(curve_1.segments)

#curve_1.seg_update([50,100],[105,50])
#print(curve_1.segments)

#print(curve_1.curve_df)
#curve_1.show_curve()

#print(curve_1.point_X)
# print(len(curve_1.segments)-1)
# print(curve_1.numbers)
# print(curve_1.steps)


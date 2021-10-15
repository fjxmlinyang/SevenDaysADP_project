class CurrModelPara():
    def __init__(self, LAC_last_windows, probabilistic, RT_DA, date, curr_day, scenario, current_stage, day_period):

        # set the length of the rolling window
        # LAC_window = 1

        # indicate if the current window is the last window, default as 0
        # LAC_last_windows = 0

        # 0:apply the deterministic forecast, 1:apply the probabilistic forecast
        #probabilistic = 0

        # read time periods

        self.LAC_last_windows = LAC_last_windows
        self.RT_DA = RT_DA
        self.probabilistic = probabilistic
        self.date = date
        self.curr_day =curr_day
        self.scenario = scenario
        self.current_stage = current_stage
        self.day_period = day_period

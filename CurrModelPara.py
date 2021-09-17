class CurrModelPara():
    def __init__(self, LAC_last_windows, probabilistic, RT_DA, date, LAC_bhour, scenario, current_stage, time_period):

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
        self.LAC_bhour = LAC_bhour
        self.scenario = scenario
        self.current_stage = current_stage
        self.time_period = time_period

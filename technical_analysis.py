import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None






class technical_analysis():
    ''' Calculate indicators and osilators of Technical Analysis

    Attributes:
    ==========

    df: The Data Frame That most be Analysed.

    col_name: calculate indicators such as moving average based on this column of df.

    Methods:
    =======

    sharp_ratio: calculate sharp ratio

    sortino_ratio: calculate sortino ratio

    moving_average: calculate moving average indicator

    macd: calculate MACD osilator (moving average convergence divergence)

    rsi: calculate RSI (Relative Strength Index)

    stochastic: calculate stochastic osilator

    bolinger_band: calculate bolinger band

    back_test: test a strategy on historical data
    '''

    def __init__(self, df=None, col_name='Adj Close'):

        self.df = df.copy()
        self.col_name = col_name

    def __repr__(self):
        return 'An Object Of Technical Analysis class'


    def sharp_ratio(self, mode = 'daily', risk_free_rate = 0):

        daily_return = self.df[self.col_name].pct_change().dropna()
        daily_mean = daily_return.mean()
        daily_std = daily_return.std()
        daily_SR = (daily_mean - risk_free_rate) / daily_std
        annualized_SR = daily_SR * np.sqrt(252)
        if mode == 'daily':
            return daily_SR
        else:
            return annualized_SR


    def sortino_ratio(self, mode='annualized', threshold = 0, risk_free_rate = 0):
        daily_return = self.df[self.col_name].pct_change().dropna()
        daily_mean = daily_return.mean()
        down_side_volatility = daily_return[daily_return < threshold].std()
        daily_sortino_r = (daily_mean - risk_free_rate) / down_side_volatility
        annualized_SR = daily_sortino_r * np.sqrt(252)
        if mode == 'annualized':
            return annualized_SR
        else:
            return daily_sortino_r



    def moving_average(self, mode='SMA', period1=20, period2 = None):
        '''
        Calculate Moving Average

        :param mode: str default "SMA"
            mode can be SMA (Simple Moving Average) OR EMA (Exponential Moving Average)
        :param period1: type int default 20
            calculate first moving average
        :param period2: type int default None
            calculate second moving average

        :return: The Data Frame Of Moving Average
        '''

        ma = pd.DataFrame()
        if mode == 'SMA':
            ma[str(period1)+mode] = self.df[self.col_name].rolling(window=period1).mean()
            if period2 is not None:
                ma[str(period2)+mode] = self.df[self.col_name].rolling(window=period2).mean()

        else:
            ma[str(period1)+mode] = self.df[self.col_name].ewm(span=period1, min_periods=period1).mean()
            if period2 is not None:
                ma[str(period2)+mode] = self.df[self.col_name].ewm(span=period2, min_periods=period2).mean()

        if period2 is None:
            return ma[str(period1)+mode]
        else:
            return ma[[str(period1)+mode, str(period2)+mode]]

    def macd(self, fast_period=12, slow_period=26, sig_period=9):
        MACD = pd.DataFrame()
        fast_ma = self.moving_average(mode='EMA', period1=fast_period).to_frame()[str(fast_period)+'EMA']
        slow_ma = self.moving_average(mode='EMA', period1=slow_period).to_frame()[str(slow_period)+'EMA']
        MACD['macd'] = fast_ma - slow_ma
        MACD['macd_signal'] = MACD.ewm(span=sig_period, min_periods=sig_period).mean()
        return MACD[['macd', 'macd_signal']]

    def rsi(self, period=14, up_threshold=70, down_threshold=30):
        high_days = np.where(self.df[self.col_name].diff() > 0, self.df[self.col_name].diff(), 0)
        low_days = np.where(self.df[self.col_name].diff() < 0, -self.df[self.col_name].diff(), 0)
        ma_h_days = pd.DataFrame(high_days).rolling(window=period).mean()
        ma_l_days = pd.DataFrame(low_days).rolling(window=period).mean()
        rsi = 100 * (ma_h_days / (ma_h_days + ma_l_days))
        rsi.set_index(self.df.index, inplace=True)
        rsi.columns = ['rsi']

        return rsi

    def stochastic(self, period_K= 14, period_D = 3, col_names= None):
        stoc = self.df.copy()
        if col_names is not None:
            stoc.rename(columns={col_names[0]:'Open', col_names[1]:'High', col_names[2]:'Low', col_names[3]:'Close'}, inplace=True)
        High = stoc['High'].rolling(period_K).max()
        Low = stoc['Low'].rolling(period_K).min()
        Close = stoc['Close']
        stoc['k_percent'] = ((Close - Low) / (High - Low)) * 100
        stoc['d_percent'] = stoc.k_percent.rolling(window=period_D).mean()
        return stoc[['k_percent', 'd_percent']]

    def bolinger_band(self, period=20, std_p=2):
        BB = self.df.copy()
        ma = BB[self.col_name].rolling(window=period).mean()
        std = BB[self.col_name].rolling(window=period).std()
        BB['BB_UP'] = ma + std_p * std
        BB['BB_DOWN'] = ma - std_p * std
        BB['BB_MA'] = ma
        return BB[['BB_UP', 'BB_DOWN', 'BB_MA']]


    def back_test(self, short_sell=True, tc_buy= None, tc_sell=None, strategy=dict()):

        testing = self.df[self.col_name].to_frame().copy()
        testing.columns = ['price']
        testing['return'] = np.log(testing.price.div(testing.price.shift(1)))
        testing['position'] = int(0)
        n = int(len(strategy))
        if 'moving_average' in strategy:
            mode, period1, period2 = strategy['moving_average']
            testing = pd.concat([testing, self.moving_average(mode=mode,period1=period1,period2=period2)], axis=1)
            testing.dropna(inplace=True)

            if short_sell:
                testing['MA_position'] = np.where(testing[str(period1)+mode] > testing[str(period2)+mode], 1, -1)
            else:
                testing['MA_position'] = np.where(testing[str(period1)+mode] > testing[str(period2)+mode], 1, 0)

            testing['position'] = testing['position'] + testing['MA_position'].apply(int)

        if 'macd' in strategy:

            fast_period, slow_period, sig_period = strategy['macd']
            testing = pd.concat([testing, self.macd(fast_period=fast_period, slow_period=slow_period, sig_period=sig_period)],axis=1)
            testing.dropna(inplace=True)

            if short_sell:
                testing['macd_position'] = np.where(testing['macd'] > testing['macd_signal'], 1, -1)
            else:
                testing['macd_position'] = np.where(testing['macd'] > testing['macd_signal'], 1, 0)

            testing['position'] = testing['macd_position'].apply(int) + testing['position']

        if 'rsi' in strategy:

            period, up_threshold, down_threshold = strategy['rsi']
            testing = pd.concat([testing, self.rsi(period=period, up_threshold=up_threshold, down_threshold=down_threshold)], axis=1)
            testing.dropna(inplace=True)

            if short_sell:
                testing['rsi_position'] = np.where(testing['rsi'] > up_threshold, -1, 0)
                testing['rsi_position'] = np.where(testing['rsi'] < down_threshold, 1, testing['rsi_position'])

            else:
                testing['rsi_position'] = np.where(testing['rsi'] < down_threshold, 1, 0)

            testing['position'] = testing['position'] + testing['rsi_position'].apply(int)

        if 'stochastic' in strategy:

            period_K, period_D = strategy['stochastic']
            testing = pd.concat([testing, self.stochastic(period_K=period_K, period_D=period_D)], axis=1)
            testing.dropna(inplace=True)

            if short_sell:
                testing['stochastic_position'] = np.where(testing['k_percent'] > testing['d_percent'], 1, -1)
            else:
                testing['stochastic_position'] = np.where(testing['k_percent'] > testing['d_percent'], 1, 0)

            testing['position'] = testing['position'] + testing['stochastic_position'].apply(int)

        if 'bolinger_band' in strategy:

            period, std_p = strategy['bolinger_band']
            testing = pd.concat([testing, self.bolinger_band(period=period, std_p=std_p)], axis=1)
            testing.dropna(inplace=True)
            testing['distance'] = testing['price'] - testing['BB_MA']

            testing['BB_position'] = np.where(testing['price'] < testing['BB_DOWN'], 1, np.nan)
            testing['BB_position'] = np.where(testing['distance'] * testing['distance'].shift(1) < 0, 0,testing['BB_position'])

            if short_sell:
                testing['BB_position'] = np.where(testing['price'] > testing['BB_UP'], -1, testing['BB_position'])

            testing['BB_position'] = testing['BB_position'].ffill().fillna(0)


            testing['position'] = testing['position'] + testing['BB_position'].apply(int)

        testing['position'] = np.where(testing['position'].abs() == n, testing['position']/n, 0)
        testing.dropna()

        testing['strategy_ret'] = testing['position'].shift() * testing['return']
        testing.dropna(inplace=True)
        testing['tc'] = 0
        testing['trade'] = testing['position'].diff().fillna(0)
        testing['strategy_ret - tc'] = testing['strategy_ret'].apply(np.exp)

        if tc_buy and tc_sell is not None:
            testing['strategy_ret - tc'][testing['trade'] > 0] = (1 - tc_buy) * testing.strategy_ret[testing['trade'] > 0].apply(np.exp)
            testing['strategy_ret - tc'][testing['trade'] < 0] = (1 - tc_sell) * testing.strategy_ret[testing['trade'] < 0].apply(np.exp)

        testing['c_strategy_ret - tc'] = testing['strategy_ret - tc'].cumprod()


        testing['c_strategy_ret'] = testing['strategy_ret'].cumsum().apply(np.exp)
        testing['c_return'] = testing['return'].cumsum().apply(np.exp)

        testing[['c_strategy_ret', 'c_strategy_ret - tc', 'c_return']].plot(figsize=(12, 8))

        return testing



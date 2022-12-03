import numpy as np
import pandas as pd
import plotly
import matplotlib.pyplot as plt
import cufflinks as cf
# cf.set_config_file(offline=True)
# cf.go_offline()
plotly.offline.init_notebook_mode()

stocks = pd.read_csv('stocks.csv', header=[0, 1], index_col=[0], parse_dates=True)

ge = stocks.swaplevel(axis=1).GE

class MovingAverage():
    def __init__(self, DataFrame):
        self.df = DataFrame

    def plot(self, period=20, sig_col='Adj Close', mode='Dynamic'):
        # plt.figure(figsize=(10, 3), dpi=200)
        # ax = self.df[sig_col].plot(label=sig_col)
        # self.df[sig_col].rolling(window=period).mean().plot(ax=ax, label=f'{period} days MV')
        # plt.legend()
        # plt.show()
        if mode == 'Dynamic':
            self.df['20 days MV'] = self.df[sig_col].rolling(window=period).mean()
            self.df['20 days MV'].plot()
            # .iplot(label=f'{period} days MV')
        else:
            self.df['20 days MV'] = self.df[sig_col].rolling(window=period).mean()
            self.df['20 days MV'].plot()

# MV = MovingAverage(ge)
# MV.plot()
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import sqlalchemy
import time
import requests
from io import StringIO
import json
from sqlalchemy import inspect
from pandas.tseries.offsets import BDay
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graphing import create_mom_plots
from data_handler import sql_dbs


with open('index_info.json', 'r') as f:
    dict_json = json.load(f)



today = datetime.now().date()
start = datetime(2010,1,1)
date_5bd_ago = today - BDay(5)
date_20bd_ago = today - BDay(20)

# =========================




### SHOULD RETURNS BE FACTORED BY 100 IF WEIGHTS ARE IN DECIMAL TERMS? dont actually store numerator and demoinator in the df
class Indicators:
    def __init__(self, market_index, ticker):
        self.market_index = market_index
        self.ticker = ticker

        df = sql_dbs[self.market_index][self.ticker][['Date', 'Close']].sort_values('Date')
        df = df.set_index('Date')

        self.data = df

    def calc_mom_indicators(
        self,
        name,
        periodicity: str,
        weights,
        vol_window=24,
        number_of_lags=12,
        target_vol=0.10,      # annualized target vol
        leverage_cap=5.0       # max allowed leverage
    ):

        # -----------------------------
        # Lagged returns
        # -----------------------------
        def calc_lagged_returns(number_of_lags=number_of_lags, periodicity=periodicity):
            if periodicity.lower() == "monthly":
                period_close = self.data['Close'].resample('M').last()
            elif periodicity.lower() == "weekly":
                period_close = self.data['Close'].resample('W').last()
            elif periodicity.lower() == "daily":
                period_close = self.data['Close'].resample('B').last()
            else:
                raise ValueError("Invalid periodicity")

            self.period = period_close.pct_change()
            self.data[f'{periodicity}_ret'] = self.period

            for lag in range(1, number_of_lags + 1):
                self.data[f'lagged_ret_{lag}'] = self.period.shift(lag).reindex(self.data.index)

            return self

        # -----------------------------
        # Momentum formula
        # -----------------------------
        def mom_formula(name: str, weights):
            self.data['numerator'] = 0.0
            self.data['sum_of_squared_weights'] = 0.0

            for i, w in enumerate(weights, start=1):
                self.data['numerator'] += self.data[f'lagged_ret_{i}'] * w
                self.data['sum_of_squared_weights'] += w ** 2

            self.data['denominator'] = (
                np.sqrt(self.data['sum_of_squared_weights']) *
                self.data["period_std_dev_t-1"]
            )

            self.data[name] = self.data['numerator'] / self.data['denominator']

        # -----------------------------
        # Z-score + buy/sell signal
        # -----------------------------
        def calc_z_scores(
            window=24,
            fast_vol_window=10,
            slow_vol_window=60,
            vol_panic_thresh=1.75,
            buy_z=1.0,
            sell_z=-1.0,
            extreme_z=-1.75
        ):
            # Momentum z-score
            self.data[f"{name}_z"] = (
                self.data[name] - self.data[name].rolling(window).mean()
            ) / self.data[name].rolling(window).std()

            # Volatility regime
            fast_vol = self.data["daily_ret"].rolling(fast_vol_window).std()
            slow_vol = self.data["daily_ret"].rolling(slow_vol_window).std()
            self.data["vol_regime"] = fast_vol / slow_vol

            # Raw momentum
            buy_raw = self.data[f"{name}_z"] > buy_z
            sell_raw = self.data[f"{name}_z"] < sell_z

            # SELL vetoes
            extreme_negative = self.data[f"{name}_z"] < extreme_z
            panic_vol = self.data["vol_regime"] > vol_panic_thresh

            buy_signal = buy_raw
            sell_signal = sell_raw & ~extreme_negative & ~panic_vol

            self.data[f"buy/sell_{name}"] = np.where(
                buy_signal, "buy",
                np.where(sell_signal, "sell", "-")
            )

        # -----------------------------
        # Position sizing with leverage cap
        # -----------------------------

        # def calc_position_size(z_cap=2.0, min_vol_scale=0.1, smooth_span=5):
        #     mom_z = self.data[f"{name}_z"]
        #     vol_regime = self.data["vol_regime"]

        #     # Base position from momentum strength (0..1)
        #     raw_position = ((mom_z + z_cap) / (2 * z_cap)).clip(0, 1)

        #     # Volatility-based leverage scaling
        #     # Target vol is annualized (default 10%), daily vol used here
        #     daily_target_vol = target_vol / np.sqrt(252)
        #     leverage = (daily_target_vol / self.data["daily_ret"].rolling(20).std()).clip(min_vol_scale, leverage_cap)

        #     scaled_position = raw_position * leverage

        #     # Smooth position to avoid whipsaw
        #     self.data[f"position_{name}"] = (
        #         scaled_position
        #         .ewm(span=smooth_span, adjust=False)
        #         .mean()
        #     )

        # # -----------------------------
        # # Strategy performance on price scale
        # # -----------------------------
        # def calc_strategy_performance():
        #     position_col = f"position_{name}"

        #     # Use lagged position to avoid look-ahead bias
        #     strat_ret = self.data[position_col].shift(1) * self.data["daily_ret"]
        #     self.data[f"strategy_ret_{name}"] = strat_ret

        #     # Initialize strategy price on same scale as Close
        #     start_price = self.data["Close"].iloc[0]
        #     self.data[f"strategy_price_{name}"] = (
        #         (1 + strat_ret.fillna(0)).cumprod() * start_price
        #     )

        # -----------------------------
        # Execution order
        # -----------------------------
        calc_lagged_returns(number_of_lags=number_of_lags, periodicity=periodicity)

        if len(weights) != number_of_lags:
            raise ValueError(f"{len(weights)} weights given — {number_of_lags} required")

        # Rolling standard deviation of lagged period returns
        period_ret_lagged = self.data[f'{periodicity}_ret'].shift(1)
        self.data['period_std_dev_t-1'] = period_ret_lagged.rolling(vol_window).std()

        # Compute momentum, z-scores, position, and strategy
        mom_formula(name, weights)
        calc_z_scores()
        # calc_position_size()
        # calc_strategy_performance()

        return self




strat_weights = {"mom_1,4":[0.25,0.25,0.25,0.25,0,0,0,0,0,0,0,0],
                 "mom_5,8":[0,0,0,0,0.25,0.25,0.25,0.25,0,0,0,0],
                 "mom_9,11":[0,0,0,0,0,0,0,0,0.334,0.333,0.333,0],

                
                 }


indicators_dbs = {}  
for market_ticker in ["DJI"]:  
    secs = list(sql_dbs[market_ticker].keys())
    indicators_dbs[market_ticker] = {}

    for sec in secs:
        ind_obj = Indicators(market_ticker, sec)
        for name, weights in strat_weights.items():
            ind_obj.calc_mom_indicators(
                name=name,
                weights=weights,
                periodicity="daily"
            )

        # store the resulting DataFrame (with all strategies' columns)
        indicators_dbs[market_ticker][sec] = ind_obj.data


all_plots_fig = create_mom_plots(indicators_dbs)





## next todo: optimise for weights and for z score, holding period - step 1 will be productionising the strat
## learn how to use vol in mom, why mom didnt work in 2025
## dict_keys(['AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT'])

""""" the (pragmatic) strategy:
    Short lookbacks (like 1–3 months) can react quickly, but may also overfit to random noise → more false signals.

    Long lookbacks (like 6–12 months) smooth out noise, but may lag the trend → you enter late, exit late.



    end up with a ts price chart a stock, say AAPL
    plot where mom indicator was buy/sell_
    give the mom_ind a z-score to do this
    only using backward looking data, of course

    ## OHV has combined transaction and slippage costs at 0.6-0.8% per year
    for ticker.data, think about getting autocorrelation matrix between lags

    test different weights on each lag across securities to optimise

Formula: mom_i = w_i * R_t-1 + 

 mom(1,4) based on the past four months’ returns (
w1  w2  w3  w4 1/ 4
, other lags zero)
 mom(5,8) based on returns from 5 to 8 months ago (
w5
 w6  w7
 w8 1/ 4
, other lags zero)
 mom(9,11) based on returns from 9 to 11 month ago (
w9  w10  w11 1/3
, other lags zero)
 momCTA, based on the past 11 month returns, weights given in Figure 4 (right panel)

"""
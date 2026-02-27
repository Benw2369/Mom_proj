import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import sqlalchemy
import time
import requests
from io import StringIO
import json
import os
from sqlalchemy import inspect
from pandas.tseries.offsets import BDay
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

from data_handler import sql_dbs

# Get the directory where this file is located
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load index info with error handling
try:
    index_info_path = os.path.join(_FILE_DIR, 'index_info.json')
    with open(index_info_path, 'r') as f:
        dict_json = json.load(f)
except FileNotFoundError:
    print("WARNING: index_info.json not found in mom_strat_new.py")
    dict_json = {"DJI": {}, "FTSE100": {}}

today = datetime.now().date()
start = datetime(2010, 1, 1)
date_5bd_ago = today - BDay(5)
date_20bd_ago = today - BDay(20)


class Indicators:
    def __init__(self, market_index, ticker):
        self.market_index = market_index
        self.ticker = ticker

        # Load and sort data with error handling
        try:
            if market_index not in sql_dbs or ticker not in sql_dbs[market_index]:
                print(f"WARNING: No data for {market_index}/{ticker}")
                self.data = pd.DataFrame({'Close': []})
            else:
                df = sql_dbs[self.market_index][self.ticker][['Date', 'Close']].copy()
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df = df.set_index('Date')
                self.data = df
        except Exception as e:
            print(f"Error loading data for {market_index}/{ticker}: {str(e)}")
            self.data = pd.DataFrame({'Close': []})
        self.plot = None

    def calc_mom_indicators(
        self,
        name,
        periodicity: str,
        weights,
        vol_window=30,
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

            # Forward-fill missing data and calculate returns
            self.period = period_close.ffill().pct_change()
            self.data[f'{periodicity}_ret'] = self.period.reindex(self.data.index)

            for lag in range(1, number_of_lags + 1):
                self.data[f'lagged_ret_{lag}'] = self.period.shift(lag).reindex(self.data.index)

        # -----------------------------
        # Momentum formula
        # -----------------------------
        def mom_formula(name: str, weights):
            self.data['numerator'] = 0.0
            self.data['sum_of_squared_weights'] = 0.0

            for i, w in enumerate(weights, start=1):
                if f'lagged_ret_{i}' not in self.data.columns:
                    self.data[f'lagged_ret_{i}'] = 0.0
                self.data['numerator'] += self.data[f'lagged_ret_{i}'].fillna(0) * w
                self.data['sum_of_squared_weights'] += w ** 2

            self.data['denominator'] = (
                np.sqrt(self.data['sum_of_squared_weights']) *
                self.data.get("period_std_dev_t-1", pd.Series(1.0, index=self.data.index))
            )

            self.data[name] = self.data['numerator'] / self.data['denominator'].replace(0, np.nan)

        # -----------------------------
        # Z-score + buy/sell signal
        # -----------------------------
        def calc_z_scores(
            window=30,
            fast_vol_window=10,
            slow_vol_window=60,
            vol_panic_thresh=1.75,
            buy_z=1.0,
            sell_z=-1.0,
            extreme_z=-1.75
        ):
            if 'daily_ret' not in self.data.columns:
                self.data['daily_ret'] = self.data['Close'].pct_change()

            self.data[f"{name}_z"] = (
                self.data[name] - self.data[name].rolling(window).mean()
            ) / self.data[name].rolling(window).std()

            fast_vol = self.data["daily_ret"].rolling(fast_vol_window).std()
            slow_vol = self.data["daily_ret"].rolling(slow_vol_window).std()
            self.data["vol_regime"] = fast_vol / slow_vol

            buy_raw = self.data[f"{name}_z"] > buy_z
            sell_raw = self.data[f"{name}_z"] < sell_z

            extreme_negative = self.data[f"{name}_z"] < extreme_z
            panic_vol = self.data["vol_regime"] > vol_panic_thresh

            buy_signal = buy_raw
            sell_signal = sell_raw & ~extreme_negative & ~panic_vol

            self.data[f"buy/sell_{name}"] = np.where(
                buy_signal, "buy",
                np.where(sell_signal, "sell", "-")
            )

            # Collapse consecutive identical signals so only the first in each run remains
            signal_series = self.data[f"buy/sell_{name}"].copy()
            duplicate_mask = signal_series.isin(["buy", "sell"]) & signal_series.eq(signal_series.shift())
            signal_series.loc[duplicate_mask] = "-"
            self.data[f"buy/sell_{name}"] = signal_series

        # -----------------------------
        # Plots
        # -----------------------------
        def create_plots():
            returns = pd.to_numeric(self.data.get('daily_ret', pd.Series()), errors='coerce').dropna()
            if returns.empty:
                self.plot = None
                return

            pos_ret = returns[returns > 0]
            neg_ret = returns[returns < 0]

            # Fit distributions safely
            mu_pos, std_pos = norm.fit(pos_ret) if not pos_ret.empty else (0, 1)
            mu_neg, std_neg = norm.fit(neg_ret) if not neg_ret.empty else (0, 1)

            skew_pos = skew(pos_ret) if not pos_ret.empty else 0
            kurt_pos = kurtosis(pos_ret, fisher=True) if not pos_ret.empty else 0
            skew_neg = skew(neg_ret) if not neg_ret.empty else 0
            kurt_neg = kurtosis(neg_ret, fisher=True) if not neg_ret.empty else 0

            x_min = returns.min() * 1.1
            x_max = returns.max() * 1.1
            x = np.linspace(x_min, x_max, 1000)

            pdf_pos = norm.pdf(x, mu_pos, std_pos)
            pdf_neg = norm.pdf(x, mu_neg, std_neg)

            fig, ax = plt.subplots(figsize=(9,5))
            ax.plot(x, pdf_pos, color='green', lw=2, label='Positive Returns')
            ax.plot(x, pdf_neg, color='red', lw=2, label='Negative Returns')

            ax.text(0.05, 0.95, f'Positive μ={mu_pos:.4f}, σ={std_pos:.4f}\nskew={skew_pos:.4f}, kurt={kurt_pos:.4f}',
                    transform=ax.transAxes, verticalalignment='top', color='green')
            ax.text(0.05, 0.75, f'Negative μ={mu_neg:.4f}, σ={std_neg:.4f}\nskew={skew_neg:.4f}, kurt={kurt_neg:.4f}',
                    transform=ax.transAxes, verticalalignment='top', color='red')

            ax.set_title(f"Normal Fit of Daily Returns: {self.ticker}")
            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Probability Density")
            ax.legend()

            self.plot = fig
            plt.close(fig)  # Close to avoid memory issues

        # -----------------------------
        # Execution
        # -----------------------------
        calc_lagged_returns(number_of_lags=number_of_lags, periodicity=periodicity)

        if len(weights) != number_of_lags:
            raise ValueError(f"{len(weights)} weights given — {number_of_lags} required")

        period_ret_lagged = self.data[f'{periodicity}_ret'].shift(1)
        self.data['period_std_dev_t-1'] = period_ret_lagged.rolling(vol_window).std()
        # Add faster 10-day volatility for plotting
        self.data['period_std_dev_t-1_10'] = period_ret_lagged.rolling(10).std()

        mom_formula(name, weights)
        calc_z_scores()
        create_plots()

        return self



## TODO next: use ~10 securities which have been in DJI since 2000 to test strategy properly



# strat_weights = {"mom_1,4":[0.25,0.25,0.25,0.25,0,0,0,0,0,0,0,0],
#                  "mom_5,8":[0,0,0,0,0.25,0.25,0.25,0.25,0,0,0,0],
#                  "mom_9,11":[0,0,0,0,0,0,0,0,0.334,0.333,0.333,0],

                
#                  }


# indicators_dbs = {}  
# for market_ticker in ["DJI"]:  
#     secs = list(sql_dbs[market_ticker].keys())
#     indicators_dbs[market_ticker] = {}

#     for sec in secs:
#         ind_obj = Indicators(market_ticker, sec)
#         for name, weights in strat_weights.items():
#             ind_obj.calc_mom_indicators(
#                 name=name,
#                 weights=weights,
#                 periodicity="daily"
#             )

#         # store the resulting DataFrame (with all strategies' columns)
#         indicators_dbs[market_ticker][sec] = ind_obj.data











# meta_prices.to_csv("large_df.csv")



# all_plots_fig = create_mom_plots(indicators_dbs)
# all_plots_fig.show()




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
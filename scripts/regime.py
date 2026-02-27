
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as web


t = [0,1,2,3,4,5]
range(len(t)-1)















class RegimeDetector:
    def __init__(self, meta_dict, version="v1", start="2010-01-01", end="2025-12-31"):
        self.meta_dict = meta_dict
        self.version = version
        self.start = start
        self.end = end
        self.data = pd.DataFrame()              # daily data
        self.returns = pd.DataFrame()           # monthly returns
        self.z_scores = pd.DataFrame()          # z-scores

    # ----------------- DATA LOADING -----------------
    def append_prices(self):
        def get_adj_close(symbol, start, end):
            # Try yfinance
            df = yf.download(symbol, start=start, end=end, progress=False)
            if not df.empty and ("Adj Close" in df.columns or "Close" in df.columns):
                series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
                print(f"{symbol}: successfully downloaded via yfinance")
                return series

            # Fallback to FRED
            try:
                df = web.DataReader(symbol, "fred", start=start, end=end)
                series = df.iloc[:, 0]
                print(f"{symbol}: successfully downloaded via FRED")

                # Convert DGS10 yield to bond price if it's the ief_bond_index
                if symbol == "DGS10":
                    series = self._dgs10_to_price(series)
                    print(f"{symbol}: converted yield to bond price")
                return series
            except Exception as e2:
                print(f"Failed to download {symbol} from both sources: {e2}")
                return pd.Series(dtype=float)

        for name, ticker in self.meta_dict.get(self.version, {}).items():
            temp = get_adj_close(ticker, self.start, self.end)
            if isinstance(temp, pd.Series):
                temp = temp.to_frame(name)
            elif isinstance(temp, pd.DataFrame):
                temp = temp.iloc[:, [0]].rename(columns={temp.columns[0]: name})
            else:
                continue
            self.data = pd.concat([self.data, temp], axis=1)

    # ----------------- HELPER FUNCTION -----------------
    def _dgs10_to_price(self, yield_series, maturity_years=10, face_value=100):
        """
        Convert 10-year Treasury yield series to approximate bond prices.
        Assumes annual coupon equal to yield (par bond).
        """
        prices = []
        N = maturity_years
        F = face_value

        for y in yield_series:
            if pd.isna(y):
                prices.append(np.nan)
                continue
            y_decimal = y / 100
            C = y_decimal * F
            discounted_coupons = sum([C / ((1 + y_decimal)**i) for i in range(1, N+1)])
            discounted_principal = F / ((1 + y_decimal)**N)
            prices.append(discounted_coupons + discounted_principal)

        return pd.Series(prices, index=yield_series.index)

    # ----------------- RESAMPLING -----------------
    @staticmethod
    def resample_monthly(data):
        return data.resample("ME").last().dropna(how="all")

    # ----------------- YIELD CURVE -----------------
    @staticmethod
    def create_yield_curve(data, long_term_ticker="10Y Yields", short_term_ticker="3m Yields"):
        data['yield_curve'] = data[long_term_ticker] - data[short_term_ticker]
        return data.drop(columns=[long_term_ticker]).dropna(how="all")

    # ----------------- RETURNS -----------------
    @staticmethod
    def convert_to_returns(data, months=12, keep_originals=False):
        orig_cols = data.columns.tolist()
        for col in orig_cols:
            data[f"{col}_ret_{months}m"] = data[col].pct_change(months)
        if keep_originals:
            return data.dropna(how="all")
        else:
            return data.drop(columns=orig_cols).dropna(how="all")

    # ----------------- ROLLING CORRELATION -----------------
    def compute_daily_corr_resample_monthly(self, stock_col, bond_col, corr_name, years=3):
        """Compute rolling correlation on daily data and map to monthly."""
        daily_returns = self.data.pct_change()
        window = 252 * years  # trading days
        daily_returns[corr_name] = daily_returns[stock_col].rolling(window).corr(daily_returns[bond_col])
        monthly_corr = daily_returns[[corr_name]].resample('ME').last()
        return monthly_corr

    # ----------------- METRICS CALCULATION -----------------
    def calc_metrics_and_resample(self, include_bond_corr=False):
        # 1. Yield curve
        self.returns = self.create_yield_curve(self.data.copy())

        # 2. Daily 3-year rolling stock-bond correlation -> monthly
        if include_bond_corr:
            monthly_corr = self.compute_daily_corr_resample_monthly(
                stock_col="US Equities (S&P)",
                bond_col="Bond Index",
                corr_name="stock_bond_corr_3y",
                years=3
            )

        # 3. Resample other columns to monthly
        self.returns = self.resample_monthly(self.returns)

        # 4. Convert to 12-month returns
        self.returns = self.convert_to_returns(self.returns)

        # 5. Merge correlation if included
        if include_bond_corr:
            self.returns = self.returns.merge(monthly_corr, left_index=True, right_index=True)

    # ----------------- Z-SCORES & METRICS -----------------
    def resample_calculate_metrics_and_z_scores(self):
        def compute_z_scores(df, clip="n"):
            df = df.copy()
            original_cols = df.columns.tolist()
            for col in df.columns:
                z_adj = df[col] / df[col].rolling(window=120).std()
                if clip == "y":
                    df[f"{col}_z_score_win"] = z_adj.clip(-3, 3)
                else:
                    df[f"{col}_z_score"] = z_adj
            df = df.drop(columns=original_cols)
            return df

        def create_means_dict(df):
            means_dict = {}
            for col in df.columns:
                means_dict[col] = {"mean": float(df[col].mean().round(5)),
                                   "std": float(df[col].std().round(5))}
            return means_dict

        def compute_euclidean_distances(df):
            dist_df = pd.DataFrame(index=df.index)
            end_date = df.index[-1]
            for col in df.columns:
                ref = df[col].loc[end_date]
                dist_df[col] = np.sqrt((df[col] - ref) ** 2)
            return dist_df

        def create_corr_matrix(self):
            corr_matrix = self.returns.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix of Returns")
            return fig

        self.z_scores = compute_z_scores(self.returns, clip="n")
        self.z_scores_clipped = compute_z_scores(self.returns, clip="y")
        self.means_dict_z_scores = create_means_dict(self.z_scores)
        self.means_dict_z_scores_clipped = create_means_dict(self.z_scores_clipped)
        self.euclidean_dists = compute_euclidean_distances(self.z_scores_clipped)
        self.correlation_of_variables = create_corr_matrix(self)

    # ----------------- PLOTTING -----------------
    def create_figures(self):
        def create_sub_plots(self):
            df = self.z_scores_clipped.copy().rename(columns= {
            "US Equities (S&P)_ret_12m_z_score_win": "US Equities z-score",
            "Oil_ret_12m_z_score_win": "Oil z-score",
            "Copper_ret_12m_z_score_win": "Copper z-score",
            "Volatility_ret_12m_z_score_win": "Volatility z-score",
            "Bond Index_ret_12m_z_score_win": "Bond Index z-score",
            "3m Yields_ret_12m_z_score_win": "3m Yields z-score",
            "yield_curve_ret_12m_z_score_win": "Yield Curve z-score",
        }
        )
            
            # Drop bond index from plotting
            df = df[[c for c in df.columns if "Bond Index" not in c]]

            # Main plot
            fig_main, ax_main = plt.subplots(figsize=(12,6))
            df.plot(ax=ax_main)
            ax_main.set_title("Time Series Plot")
            ax_main.set_xlabel("Date")
            ax_main.set_ylabel("Euclidean Distance")
            fig_main.tight_layout()

            # Subplots per column
            num_cols = len(df.columns)
            fig_sub, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(12, 3*num_cols), sharex=False)
            if num_cols == 1:
                axes = [axes]

            for i, col in enumerate(df.columns):
                axes[i].plot(df.index, df[col], label=col, color="blue")
                axes[i].set_title(col)
                axes[i].grid(True)
                axes[i].legend()
                axes[i].set_ylabel("Value")
            axes[-1].set_xlabel("Date")
            fig_sub.tight_layout()

            return fig_main, fig_sub

        self.main_fig, self.sub_plots_fig = create_sub_plots(self)


    
start_date = "1970-01-01"
end_date = "2025-12-31"

meta_dict = {
    "v1": {
        "US Equities (S&P)": "^GSPC",
        "Oil": "WTISPLC",
        "Copper": "WPUSI019011",
        "Volatility": "VXOCLS",
        "Bond Index": "DGS10",
        "10Y Yields": "DGS10",
        "3m Yields": "DTB3"
    }
}

regime = RegimeDetector(meta_dict, version="v1", start=start_date, end=end_date)

regime.append_prices()
regime.calc_metrics_and_resample()
regime.resample_calculate_metrics_and_z_scores()
regime.create_figures()


df = regime.data.fillna(0)
nonzero_counts = {}
for col in df.columns:
    temp = df[[col]]
    temp = temp[temp[col] != 0]
    nonzero_counts[col] = len(temp)
nonzero_counts


regime.correlation_of_variables
regime.main_fig
regime.data
regime.returns
regime.z_scores
regime.euclidean_dists


regime.z_scores_clipped
regime.means_dict_z_scores
regime.means_dict_z_scores_clipped
dir(regime)
regime.sub_plots_fig













# ----------------- Prepare euc_df -----------------
euc_df = regime.euclidean_dists.dropna()
euc_df["Global Distance"] = euc_df.sum(axis=1)

clean_cols = {
    "US Equities (S&P)_ret_12m_z_score_win": "US Equities",
    "Oil_ret_12m_z_score_win": "Oil",
    "Copper_ret_12m_z_score_win": "Copper",
    "Volatility_ret_12m_z_score_win": "Volatility",
    "Bond Index_ret_12m_z_score_win": "Bond Index",
    "3m Yields_ret_12m_z_score_win": "3m Yields",
    "yield_curve_ret_12m_z_score_win": "Yield Curve",
    "stock_bond_corr_3y_z_score_win": "Stock-Bond Corr",
}
euc_df = euc_df.rename(columns=clean_cols)
euc_df.index = pd.to_datetime(euc_df.index)
euc_df["Global Distance Percentile"] = euc_df["Global Distance"].rank(pct=True) * 100

# ----------------- Slice from 2008 onward -----------------
start_date = pd.Timestamp("2008-01-01")
euc_df = euc_df[euc_df.index >= start_date]

# ----------------- Shaded regions -----------------
bottom_20 = euc_df["Global Distance Percentile"] <= 20
recent_start = euc_df.index[-1] - pd.DateOffset(years=3)
recent_3yrs = euc_df.index >= recent_start

green_mask = bottom_20 & ~recent_3yrs
grey_mask = recent_3yrs  # grey overrides green

# ----------------- Download S&P 500 prices and compute monthly returns -----------------




# ----------------- Prepare euc_df -----------------
euc_df = regime.euclidean_dists.dropna()
euc_df["Global Distance"] = euc_df.sum(axis=1)
euc_df.index = pd.to_datetime(euc_df.index)
euc_df["Global Distance Percentile"] = euc_df["Global Distance"].rank(pct=True) * 100

# Slice from 2008 onward
start_date = pd.Timestamp("2008-01-01")
end_date = euc_df.index[-1]
euc_df = euc_df[euc_df.index >= start_date]

# Shaded regions
bottom_20 = euc_df["Global Distance Percentile"] <= 20
recent_start = euc_df.index[-1] - pd.DateOffset(years=3)
recent_3yrs = euc_df.index >= recent_start

green_mask = bottom_20 & ~recent_3yrs
grey_mask = recent_3yrs  # grey overrides green


# ----------------- Fama-French Momentum Factor -----------------
# 1927 to present








def get_mom_data():
    # Correct URL for momentum TXT data
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_TXT.zip"

    r = requests.get(url)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Usually one .txt file inside
        txt_name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        with z.open(txt_name) as f:
            lines = f.read().decode('utf-8').splitlines()

    # Locate header row (first date entry like 192701)
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("192"))
    data_lines = []
    for line in lines[start:]:
        if "Copyright" in line or not line.strip():
            break
        data_lines.append(line)

    # Create DataFrame
    df = pd.DataFrame([l.split() for l in data_lines])
    df.columns = ["Date", "MOM"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce") / 100

    mom = df.set_index("Date")["MOM"].to_frame()
    mom = df.set_index("Date")[["MOM"]].copy()
    mom.rename(columns={"MOM": "Momentum return"}, inplace=True)

    # Create the cumulative index starting at 100
    mom["Index Close"] = 100 * (1 + mom["Momentum return"]).cumprod()
    return mom

mom = get_mom_data()








def plot_mom_vs_dist():
    mom = get_mom_data()

    mom = mom[mom.index>"2008-01-01"]
    # ----------------- Prepare series -----------------
    mom_ret_series = mom["Momentum return"]  # monthly returns
    mom_log_series = np.log(mom["Index Close"])  # log of cumulative index

    green_mask_mom = pd.Series(green_mask, index=euc_df.index).reindex(mom.index, method="ffill").fillna(False)
    grey_mask_mom = pd.Series(grey_mask, index=euc_df.index).reindex(mom.index, method="ffill").fillna(False)


    # ----------------- Plot -----------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(24, 24))

    # --- Panel 1: Global Distance ---
    ax1.plot(euc_df.index, euc_df["Global Distance"], color="red", linewidth=1.8, label="Global Distance")
    ax1.fill_between(euc_df.index,
                    euc_df["Global Distance"].min(),
                    euc_df["Global Distance"].max(),
                    where=green_mask,
                    color="green", alpha=0.25, label="Bottom 20%")
    ax1.fill_between(euc_df.index,
                    euc_df["Global Distance"].min(),
                    euc_df["Global Distance"].max(),
                    where=grey_mask,
                    color="grey", alpha=0.2, label="Most Recent 3 Years")
    ax1.set_ylabel("Aggregate Similarity")
    ax1.set_title("Aggregate Similarity with Regime Highlighting")
    ax1.grid(True)
    ax1.legend()
    ax1.invert_yaxis()

    # --- Panel 2: Momentum Index Monthly Returns ---
    ax2.bar(mom_ret_series.index[mom_ret_series >= 0],
            mom_ret_series[mom_ret_series >= 0],
            color="blue", width=20, alpha=0.7, label="Positive Return")
    ax2.bar(mom_ret_series.index[mom_ret_series < 0],
            mom_ret_series[mom_ret_series < 0],
            color="red", width=20, alpha=0.7, label="Negative Return")

    # Shaded regimes
    ax2.fill_between(mom_ret_series.index,
                    mom_ret_series.min(),
                    mom_ret_series.max(),
                    where=green_mask_mom,
                    color="green", alpha=0.25)
    ax2.fill_between(mom_ret_series.index,
                    mom_ret_series.min(),
                    mom_ret_series.max(),
                    where=grey_mask_mom,
                    color="grey", alpha=0.2)

    ax2.set_ylabel("Momentum Monthly Return")
    ax2.set_title("US Long Momentum Index - Monthly Returns")
    ax2.grid(True)
    ax2.legend()

    # --- Panel 3: Momentum Index Log Prices ---
    ax3.plot(mom_log_series.index, mom_log_series, color="purple", linewidth=1.8, label="Log Prices")

    # Shaded regimes
    ax3.fill_between(mom_log_series.index,
                    mom_log_series.min(),
                    mom_log_series.max(),
                    where=grey_mask_mom,
                    color="grey", alpha=0.2)
    
    ax3.fill_between(mom_log_series.index,
                    mom_log_series.min(),
                    mom_log_series.max(),
                    where=green_mask_mom,
                    color="green", alpha=0.25)
    
    ax3.set_ylabel("Log Price")
    ax3.set_title("US Long - Momentum Index Log Prices")
    ax3.grid(True)
    ax3.legend()

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

plot_mom_vs_dist()

def plot_euclidean_components(euc_df, green_mask, grey_mask):
    """
    Plot subplots for all Euclidean Distance components except the final two columns,
    with regime shading.
    """
    # Drop & rename as before
    euc_df_plot = euc_df.drop(columns=["Bond Index_ret_12m_z_score_win"], errors='ignore')
    euc_df_plot = euc_df_plot.rename(columns={"3m Yields": "Monetary Policy (3m Yields)"}, errors='ignore')

    cols_to_plot = euc_df_plot.columns[:-2]  # drop last two
    n = len(cols_to_plot)

    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(20, 3.5*n))

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_to_plot):
        ax.plot(euc_df_plot.index, euc_df_plot[col], linewidth=1.6)
        # Shaded regimes

        ax.fill_between(euc_df_plot.index,
                        euc_df_plot[col].min(),
                        euc_df_plot[col].max(),
                        where=grey_mask,
                        color="grey", alpha=0.2)
        
        ax.fill_between(euc_df_plot.index,
                        euc_df_plot[col].min(),
                        euc_df_plot[col].max(),
                        where=green_mask,
                        color="green", alpha=0.25)
    
        ax.set_title(col)
        ax.grid(True)
        ax.invert_yaxis()  # if desired

    axes[-1].set_xlabel("Date")
    fig.suptitle("Euclidean Distance Components", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()

# Call it
plot_euclidean_components(euc_df, green_mask, grey_mask)

"""

I thought you would like to see this recent paper produced by Man AHL - i think a similar model could be valuable to you in a few ways - one of which being weighting RAG/ Trend signals depending on regime


their strategy then goes long farma-french factors with positive returns in the shaded reigons. 
You mentioned you'd like to weight your RAG and Trend indicators dynamically - this model could use momentum factor returns to do so
- a higher momentum weight when it has performed positively in similar regimes, with a heavier weighting on more recent time periods

or if you have data on the success of your signals (else we put one together) - these can be used instead of the momentum factor to produce a 'regime-aware' indicator.

These are the 7 economic varibles the paper uses which return positive alpha:

- Oil Price
- Copper Price
- Monetary Policy (3m yields)
- Yield Curve (10y - 3m yields)
- Volatility (VIX)
- S&P Prices
- Stock-Bond Correlation

could look to improve with new variables (different equity valuation ratios, inflation, credit risk)
the criteria being; lowly correlated with other variables, auto-correlation diminishing after 12 months




"""
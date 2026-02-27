import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

# Assuming you have a function that creates the plots, e.g. `create_mom_plots`
from mom_strat_new import Indicators


app = dash.Dash(__name__)

# Example indicators_dbs data
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



def create_mom_plots(indicators_dbs):
    # --- Create figure without shared x-axes to keep dropdown stable ---
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,  # important for stability with dropdown
        vertical_spacing=0.025,
        subplot_titles=("Close", "daily_ret", "mom_1,4_z", "period_std_dev_t-1")
    )

    secs = list(indicators_dbs["DJI"].keys())
    traces_per_sec = 6  # close, daily_ret, mom, std, buy, sell

    # --- Precompute x-axis ranges for last 250 rows ---
    x_ranges = {}
    for sec, df in indicators_dbs["DJI"].items():
        df_plot = df.tail(250)
        x_ranges[sec] = [df_plot.index.min(), df_plot.index.max()]

    # --- Add traces for each sec ---
    for i, (sec, df) in enumerate(indicators_dbs["DJI"].items()):
        df_plot = df.tail(250)
        visible = i == 0  # only first sec visible initially

        # Close
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["Close"], name=f"{sec} Close", visible=visible
        ), row=1, col=1)

        # daily_ret
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["daily_ret"], name=f"{sec} daily_ret", visible=visible
        ), row=2, col=1)

        # mom_1,4_z
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["mom_1,4_z"], name=f"{sec} mom_1,4_z", visible=visible
        ), row=3, col=1)

        # period_std_dev_t-1
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["period_std_dev_t-1"], name=f"{sec} period_std_dev_t-1", visible=visible
        ), row=4, col=1)

        # Buy signals
        buy_points = df_plot[df_plot["buy/sell_mom_1,4"] == "buy"]
        fig.add_trace(go.Scatter(
            x=buy_points.index, y=buy_points["Close"], mode="markers",
            marker=dict(size=10, color="green", symbol="triangle-up"),
            name=f"{sec} Buy", visible=visible
        ), row=1, col=1)

        # Sell signals
        sell_points = df_plot[df_plot["buy/sell_mom_1,4"] == "sell"]
        fig.add_trace(go.Scatter(
            x=sell_points.index, y=sell_points["Close"], mode="markers",
            marker=dict(size=10, color="red", symbol="triangle-down"),
            name=f"{sec} Sell", visible=visible
        ), row=1, col=1)

    # --- Horizontal z-score lines on subplot 3 ---
    z_levels = {1: 0.3, 1.25: 0.4, 1.5: 0.5, 1.75: 0.6,
                -1: 0.3, -1.25: 0.4, -1.5: 0.5, -1.75: 0.6}

    for z, opacity in z_levels.items():
        fig.add_hline(y=z, line_dash="dash", row=3, col=1, opacity=opacity)

    # --- Initial x-axis ranges ---
    first_sec = secs[0]
    for row in range(1, 5):
        fig.update_xaxes(range=x_ranges[first_sec], row=row, col=1)

    # --- Force tick labels only on bottom subplot ---
    for row in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=row, col=1)
    fig.update_xaxes(showticklabels=True, row=4, col=1)

    # --- Dropdown buttons ---
    buttons = []
    total_traces = len(fig.data)
    for i, sec in enumerate(secs):
        visibility = [False] * total_traces
        start = i * traces_per_sec
        end = start + traces_per_sec
        for j in range(start, end):
            visibility[j] = True

        buttons.append(dict(
            label=sec,
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": f"Stacked mom_1,4 Plots — {sec}",
                    "xaxis":  {"range": x_ranges[sec]},
                    "xaxis2": {"range": x_ranges[sec]},
                    "xaxis3": {"range": x_ranges[sec]},
                    "xaxis4": {"range": x_ranges[sec]},
                }
            ]
        ))

    # --- Layout ---
    fig.update_layout(
        width=1000,
        height=1000,
        title=f"Stacked mom_1,4 Plots — {first_sec}",
        showlegend=True,
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0, y=1.05,
            xanchor="left",
            yanchor="bottom",
            showactive=True
        )]
    )

    return fig

# =========================
# Create Dash App
# =========================

app.layout = html.Div([
    html.H1("Momentum Strategy Plots"),
    dcc.Graph(
        id='momentum-plots',
        figure=create_mom_plots(indicators_dbs)  # Pass your data here
    )
])

# =========================
# Run the app
# =========================
if __name__ == "__main__":
    app.run(debug=True)

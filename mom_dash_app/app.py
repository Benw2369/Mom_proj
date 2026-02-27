import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from mom_strat_new import Indicators
from data_handler import sql_dbs


##### ADD A LOOKBACK PERIOUD TO INDICATORS CALCULATION #####
##### ADD DAYS OF GAINS VS DAYS OF LOSSES METRIC #####

strat_weights = {
    "mom_1,4":  [0.25,0.25,0.25,0.25,0,0,0,0,0,0,0,0],
    "mom_5,8":  [0,0,0,0,0.25,0.25,0.25,0.25,0,0,0,0],
    "mom_9,11": [0,0,0,0,0,0,0,0,0.334,0.333,0.333,0]
}

indicators_cache = {}

def get_indicator_data(market_index, ticker):
    """Lazy-load indicator data on-demand to avoid startup timeout."""
    cache_key = f"{market_index}_{ticker}"
    if cache_key in indicators_cache:
        return indicators_cache[cache_key]
    
    try:
        ind = Indicators(market_index, ticker)
        for name, w in strat_weights.items():
            ind.calc_mom_indicators(name=name, weights=w, periodicity="daily")
        data = ind.data
        indicators_cache[cache_key] = data
        return data
    except Exception as e:
        print(f"Error loading indicators for {market_index}/{ticker}: {str(e)}")
        return None


indicators_dbs = {
    k: [t for t, df in sql_dbs.get(k, {}).items() if isinstance(df, pd.DataFrame) and not df.empty]
    for k in sql_dbs.keys()
}


def pdf_traces(df, label):
    # Ensure numeric
    returns = pd.to_numeric(df["daily_ret"], errors="coerce").dropna()
    if len(returns) < 2:
        return []

    # Separate positive and negative returns
    pos = returns[returns > 0]
    neg = returns[returns < 0]

    # Full x-axis covering both positive and negative returns
    x_min = returns.min() * 1.2
    x_max = returns.max() * 1.2
    x = np.linspace(x_min, x_max, 1000)

    traces = []

    # Positive distribution
    if len(pos) > 5:
        mu, sd = norm.fit(pos)
        pdf_pos = norm.pdf(x, mu, sd)
        traces.append(go.Scatter(
            x=x, y=pdf_pos,
            fill='tozeroy',
            line=dict(color='green', width=2),
            name=f"{label} Positive",
            hovertemplate=f"μ={mu:.4f}<br>σ={sd:.4f}<br>skew={skew(pos):.2f}<br>kurt={kurtosis(pos):.2f}",
            opacity=0.3
        ))

    # Negative distribution
    if len(neg) > 5:
        mu, sd = norm.fit(neg)
        pdf_neg = norm.pdf(x, mu, sd)
        traces.append(go.Scatter(
            x=x, y=pdf_neg,
            fill='tozeroy',
            line=dict(color='red', width=2),
            name=f"{label} Negative",
            hovertemplate=f"μ={mu:.4f}<br>σ={sd:.4f}<br>skew={skew(neg):.2f}<br>kurt={kurtosis(neg):.2f}",
            opacity=0.3
        ))

    return traces


# ========================= plotly figure  =========================

def create_figure(df, sec):
    fig = make_subplots(
        rows=5, cols=1,
        vertical_spacing=0.0375,
        subplot_titles=[
            "Close Price",
            "Daily Returns",
            "Momentum z-score (1,4)",
            "Daily Vol Past 30 vs 10 Days",
            "Distribution of Daily Returns"
        ]
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["daily_ret"], name="daily_ret"), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["mom_1,4_z"], name="mom_1,4_z"), 3, 1)

    vol_30 = df.get("period_std_dev_t-1")
    vol_10 = df.get("period_std_dev_t-1_10")
    if vol_30 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=vol_30, name="Vol 30d", line=dict(color="royalblue")), 4, 1)
    if vol_10 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=vol_10, name="Vol 10d", line=dict(color="orange")), 4, 1)

    buy = df[df["buy/sell_mom_1,4"] == "buy"]
    sell = df[df["buy/sell_mom_1,4"] == "sell"]

    fig.add_trace(go.Scatter(
        x=buy.index, y=buy["Close"],
        mode="markers",
        marker=dict(color="green", symbol="triangle-up", size=14),
        name="Positive Signal"
    ), 1, 1)

    fig.add_trace(go.Scatter(
        x=sell.index, y=sell["Close"],
        mode="markers",
        marker=dict(color="red", symbol="triangle-down", size=14),
        name="Negative Signal"
    ), 1, 1)

    for t in pdf_traces(df, sec):
        fig.add_trace(t, 5, 1)

    for z in [1,1.25,1.5,1.75,-1,-1.25,-1.5,-1.75]:
        fig.add_hline(y=z, row=3, col=1, line_dash="dash", opacity=0.4)

    x_range = [df.index.min(), df.index.max()]
    for r in range(1,5):
        fig.update_xaxes(range=x_range, row=r, col=1)
    for r in range(1,4):
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    fig.update_layout(
        template="plotly_white",
        height=1400,
        margin=dict(l=80, r=80, t=120, b=60),
        title=f"{sec}"
    )

    return fig

# ========================= DASH APP =========================

app = dash.Dash(__name__)

markets = list(indicators_dbs.keys())
default_market = markets[0] if markets else "DJI"
default_stocks = indicators_dbs[default_market] if default_market in indicators_dbs else []
default_stock = "AMGN" if "AMGN" in default_stocks else (default_stocks[0] if default_stocks else "AAPL")

app.layout = html.Div([
    html.H1("Short Term Momentum Signal Dashboard", style={"textAlign":"center"}),

    html.Div([
        # Sidebar
        html.Div([
            html.H3("Index"),
            dcc.Dropdown(
                id="index-dropdown",
                options=[{"label": i, "value": i} for i in markets],
                value=default_market,
                clearable=False
            ),
            html.Br(),
            html.H3("Ticker"),
            dcc.Dropdown(id="stock-dropdown", clearable=False),
            html.Br(),
            html.H3("Start Date"),
            dcc.DatePickerSingle(
                id="start-date-picker",
                display_format='YYYY-MM-DD',
                clearable=False
            ),
            html.Br(),
            html.Div(id="gain-loss-stats", style={"fontSize": "16px", "fontWeight": "bold"}),
            html.Br(),
            # Below is the added text
            html.P(
                """Mom (1,4): Based off equally weighted returns over past 1 to 4 days.
                When z-score > 1.5: buy signal.
                When z-score < -1.5: sell signal.""",
                style={"fontSize": "14px", "color": "#333"}
            )
        ], style={
            "width": "20%",
            "padding": "20px",
            "backgroundColor": "#f5f5f5",
            "borderRadius": "10px"
        }),

        # Graph
        html.Div([
            dcc.Graph(id="main-graph")
        ], style={"width": "75%", "paddingLeft": "30px"})
    ], style={"display":"flex", "justifyContent":"center", "width":"95%", "margin":"auto"})
])

# =========================

@app.callback(
    Output("stock-dropdown", "options"),
    Output("stock-dropdown", "value"),
    Input("index-dropdown", "value")
)
def update_stock_dropdown(index):
    if index not in indicators_dbs or not indicators_dbs[index]:
        return [], None
    stocks = indicators_dbs[index]  # Now a list, not a dict
    default = "AMGN" if "AMGN" in stocks else (stocks[0] if stocks else None)
    return [{"label":s, "value":s} for s in stocks], default

@app.callback(
    Output("start-date-picker", "min_date_allowed"),
    Output("start-date-picker", "max_date_allowed"),
    Output("start-date-picker", "date"),
    Input("index-dropdown", "value"),
    Input("stock-dropdown", "value")
)
def update_date_picker(index, stock):
    if not index or not stock or index not in indicators_dbs or stock not in indicators_dbs[index]:
        return None, None, None
    df = get_indicator_data(index, stock)
    if df is None or df.empty:
        return None, None, None
    
    min_date = df.index.min()
    max_date = df.index.max()
    
    # Default to one year ago from max_date
    from datetime import timedelta
    one_year_ago = max_date - timedelta(days=365)
    default_date = max(one_year_ago, min_date)  # Don't go before min_date
    
    return min_date, max_date, default_date

@app.callback(
    Output("main-graph", "figure"),
    Output("gain-loss-stats", "children"),
    Input("index-dropdown", "value"),
    Input("stock-dropdown", "value"),
    Input("start-date-picker", "date")
)
def update_graph(index, stock, start_date):
    if not index or not stock or index not in indicators_dbs or stock not in indicators_dbs[index]:
        return go.Figure().add_annotation(
            text="No data available. Please check your database connection.",
            showarrow=False, font=dict(size=20)
        ), ""
    df = get_indicator_data(index, stock)
    if df is None:
        return go.Figure().add_annotation(
            text="Error loading indicator data.",
            showarrow=False, font=dict(size=20)
        ), ""
    
    # Filter dataframe based on start_date
    if start_date:
        df_filtered = df[df.index >= start_date]
    else:
        df_filtered = df
    
    # Calculate gain/loss days
    if "daily_ret" in df_filtered.columns:
        returns = pd.to_numeric(df_filtered["daily_ret"], errors="coerce").dropna()
        days_gains = len(returns[returns > 0])
        days_losses = len(returns[returns < 0])
        stats_text = html.Div([
            html.P(f"Days with Gains: {days_gains}", style={"color": "green", "margin": "5px 0"}),
            html.P(f"Days with Losses: {days_losses}", style={"color": "red", "margin": "5px 0"})
        ])
    else:
        stats_text = ""
    
    return create_figure(df_filtered, stock), stats_text

# =========================
server = app.server

if __name__ == "__main__":
    app.run(debug=True)

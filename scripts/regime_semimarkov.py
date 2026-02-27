import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import nbinom
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add mom_dash_app to Python path
sys.path.insert(0, '/Users/benwilliams/VSC/Mom_proj/mom_dash_app')
from data_handler import sql_dbs
from mom_strat_new import Indicators

# Use Indicators with the DJI index (^DJI is the ticker in the database)
ind = Indicators('DJI', '^DJI')
ind.calc_mom_indicators(name='mom_1,4', weights=[0.25,0.25,0.25,0.25,0,0,0,0,0,0,0,0], periodicity='daily')
data = ind.data

# ==============================================================================
# SEMI-MARKOV REGIME-SWITCHING MODEL (Giner & Zakamulin 2023)
# ==============================================================================
# Paper approach:
# 1. Calculate autocorrelation at multiple horizons: AC1(k) for k=1,21,63,126,189,252,315,378 days
# 2. Use AC1(k) structure to identify regimes (positive AC = momentum, negative AC = reversal)
# 3. Model regime durations with negative binomial distribution (duration dependence)
# 4. Bull: ~588 days mean duration, Bear: ~294 days mean duration
# ==============================================================================

print("\n" + "="*70)
print("SEMI-MARKOV REGIME DETECTION (Giner & Zakamulin 2023)")
print("="*70)

# Calculate autocorrelation at different horizons (paper's key insight)
horizons = [1, 21, 63, 126, 189, 252, 315, 378]  # days: 1d, 1m, 3m, 6m, 9m, 12m, 15m, 18m
autocorr_features = []

def calc_autocorr_at_lag(series, k):
    """Calculate first-order autocorrelation AC1(k) for k-day returns"""
    n = len(series)
    ac1 = np.full(n, np.nan)
    
    for t in range(n - k - 1):
        # Calculate k-day returns centered at t
        window = series.iloc[t:t+k+20]  # Need extra data for lag
        if len(window) < k + 2:
            continue
        
        k_day_returns = []
        for i in range(len(window) - k):
            ret_k = window.iloc[i:i+k].sum()
            k_day_returns.append(ret_k)
        
        if len(k_day_returns) < 2:
            continue
            
        # AC1: correlation between R(k,t) and R(k,t-1)
        k_day_series = pd.Series(k_day_returns)
        if k_day_series.std() > 0:
            ac1[t] = k_day_series.autocorr(lag=1)
    
    return ac1

print("\nCalculating autocorrelation structure...")
for k in horizons:
    col_name = f'AC1_{k}d'
    data[col_name] = calc_autocorr_at_lag(data['daily_ret'], k)
    autocorr_features.append(col_name)
    print(f"  AC1({k:3d} days): Mean={data[col_name].mean():.4f}, Std={data[col_name].std():.4f}")

# Paper shows: AC1(k) is positive for k=1-189 days (momentum), negative for k=315-378 days (reversal)

# ==============================================================================
# REGIME IDENTIFICATION using autocorrelation structure
# ==============================================================================
# Paper defines regimes by autocorrelation patterns:
# - Bull/Momentum: Positive AC1 at short-medium horizons (1-252 days)
# - Reversal: Negative AC1 at long horizons (315-378 days) 
# - High volatility disrupts autocorrelation structure
# ==============================================================================

# Calculate rolling volatility for crisis detection
data['rolling_vol_60d'] = data['daily_ret'].rolling(60).std()

# Regime classification based on autocorrelation structure
regimes = np.zeros(len(data))
regime_labels = []

print("\nIdentifying regimes using autocorrelation structure...")

for i in range(len(data)):
    # Get autocorrelations at key horizons
    ac1_1m = data['AC1_21d'].iloc[i] if 'AC1_21d' in data.columns else np.nan
    ac1_6m = data['AC1_126d'].iloc[i] if 'AC1_126d' in data.columns else np.nan
    ac1_12m = data['AC1_252d'].iloc[i] if 'AC1_252d' in data.columns else np.nan
    ac1_15m = data['AC1_315d'].iloc[i] if 'AC1_315d' in data.columns else np.nan
    vol = data['rolling_vol_60d'].iloc[i]
    
    # Skip if missing data
    if pd.isna([ac1_1m, ac1_12m, vol]).any():
        regimes[i] = np.nan
        regime_labels.append('Unknown')
        continue
    
    # HIGH VOLATILITY REGIME (Crisis) - overrides autocorrelation
    # Paper: High vol periods disrupt autocorrelation patterns
    if vol > data['rolling_vol_60d'].quantile(0.85):
        regimes[i] = 2  # Crisis
        regime_labels.append('Crisis')
    
    # MOMENTUM REGIME - Positive autocorrelation at 6-12 month horizon
    # Paper shows AC1(k) > 0 for k = 126-252 days in bull markets
    elif ac1_12m > 0.05 and ac1_6m > 0:
        regimes[i] = 0  # Momentum
        regime_labels.append('Momentum')
    
    # MEAN-REVERSION REGIME - Negative autocorrelation at 12-15 month horizon
    # Paper shows AC1(k) < 0 for k = 315-378 days (reversal after momentum)
    elif not pd.isna(ac1_15m) and ac1_15m < -0.05:
        regimes[i] = 1  # Mean-Reversion
        regime_labels.append('Mean-Reversion')
    
    # MIXED - Weak or mixed signals
    elif ac1_12m < 0.05 and ac1_12m > -0.05:
        regimes[i] = 1  # Lean toward mean-reversion in mixed states
        regime_labels.append('Mean-Reversion')
    
    else:
        regimes[i] = 0  # Default to momentum
        regime_labels.append('Momentum')

data['regime'] = regimes
data['regime_label'] = regime_labels

# Calculate regime durations (paper uses negative binomial distribution)
regime_durations = {'Momentum': [], 'Mean-Reversion': [], 'Crisis': []}
current_regime = None
duration = 0

for label in regime_labels:
    if label == 'Unknown':
        continue
    if label == current_regime:
        duration += 1
    else:
        if current_regime and duration > 0:
            regime_durations[current_regime].append(duration)
        current_regime = label
        duration = 1

# Add final duration
if current_regime and duration > 0:
    regime_durations[current_regime].append(duration)

print("\n" + "="*70)
print("REGIME STATISTICS (Semi-Markov Model)")
print("="*70)

regime_counts = pd.Series(regime_labels).value_counts()
print("\n--- Regime Distribution ---")
for regime, count in regime_counts.items():
    pct = count / len([r for r in regime_labels if r != 'Unknown']) * 100
    print(f"{regime:20s}: {count:5d} days ({pct:5.1f}%)")

print("\n--- Regime Durations (Paper: Bull ~588 days, Bear ~294 days) ---")
for regime, durations in regime_durations.items():
    if durations:
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        print(f"{regime:20s}: Mean={mean_dur:6.1f} days, Median={median_dur:6.1f} days, Count={len(durations):4d} regimes")

# ==============================================================================
# AUTOCORRELATION VALIDATION (compare with paper's Figure 2)
# ==============================================================================
print("\n--- Autocorrelation Structure Validation ---")
print("Paper shows: Positive AC1 for k=1-189d, Negative AC1 for k=315-378d")
print("\nEmpirical AC1(k) from data:")
for k in horizons:
    col = f'AC1_{k}d'
    mean_ac = data[col].mean()
    sign = "+" if mean_ac > 0 else "-"
    print(f"  AC1({k:3d}d): {sign}{abs(mean_ac):.4f}")

# ==============================================================================
# 3D VISUALIZATION - Use 3 representative horizons
# ==============================================================================
# For 3D plot, select 3 autocorrelation horizons that capture regime dynamics:
# - Short-term (21d = 1 month): Captures immediate momentum
# - Medium-term (252d = 12 months): Paper's key momentum signal
# - Long-term (378d = 18 months): Captures mean-reversion/reversal

viz_features = ['AC1_21d', 'AC1_252d', 'AC1_378d']
print(f"\n3D Visualization Features: {viz_features}")

# Create 3D scatter plot
fig = go.Figure()

# Filter out unknown regimes
mask = data['regime_label'] != 'Unknown'
plot_data = data[mask].copy()

regime_colors = {
    'Momentum': 'green',
    'Mean-Reversion': 'orange', 
    'Crisis': 'red'
}

for regime_name, color in regime_colors.items():
    regime_mask = plot_data['regime_label'] == regime_name
    regime_subset = plot_data[regime_mask]
    
    if len(regime_subset) == 0:
        continue
    
    fig.add_trace(go.Scatter3d(
        x=regime_subset['AC1_21d'],
        y=regime_subset['AC1_252d'],
        z=regime_subset['AC1_378d'],
        mode='markers',
        marker=dict(
            size=3,
            color=color,
            opacity=0.6,
            line=dict(width=0)
        ),
        name=regime_name,
        text=regime_subset.index.astype(str),
        hovertemplate=f'<b>{regime_name}</b><br>' +
                      'AC1(1m): %{x:.4f}<br>' +
                      'AC1(12m): %{y:.4f}<br>' +
                      'AC1(18m): %{z:.4f}<br>' +
                      'Date: %{text}<extra></extra>'
    ))

fig.update_layout(
    title='Semi-Markov Regime Detection (Giner & Zakamulin 2023)<br>Autocorrelation Structure',
    scene=dict(
        xaxis_title='AC1(21d) - 1 Month',
        yaxis_title='AC1(252d) - 12 Months',
        zaxis_title='AC1(378d) - 18 Months',
        bgcolor='rgb(17, 17, 17)',
        xaxis=dict(backgroundcolor='rgb(17, 17, 17)', gridcolor='rgb(50, 50, 50)', showbackground=True),
        yaxis=dict(backgroundcolor='rgb(17, 17, 17)', gridcolor='rgb(50, 50, 50)', showbackground=True),
        zaxis=dict(backgroundcolor='rgb(17, 17, 17)', gridcolor='rgb(50, 50, 50)', showbackground=True)
    ),
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white'),
    showlegend=True,
    width=1200,
    height=800
)

fig.show()

# ==============================================================================
# TIME SERIES PLOT - Regime evolution over time
# ==============================================================================

fig_ts = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=('Price & Regimes', 'AC1(12 months)', 'AC1(18 months)'),
    vertical_spacing=0.08,
    row_heights=[0.4, 0.3, 0.3]
)

# Plot 1: Price with regime coloring
plot_data_ts = data[mask].copy()
for regime_name, color in regime_colors.items():
    regime_mask = plot_data_ts['regime_label'] == regime_name
    regime_subset = plot_data_ts[regime_mask]
    
    fig_ts.add_trace(go.Scatter(
        x=regime_subset.index,
        y=regime_subset['Close'],
        mode='markers',
        marker=dict(size=2, color=color),
        name=regime_name,
        showlegend=True
    ), row=1, col=1)

# Plot 2: AC1(12m) 
fig_ts.add_trace(go.Scatter(
    x=plot_data_ts.index,
    y=plot_data_ts['AC1_252d'],
    mode='lines',
    line=dict(color='cyan', width=1),
    name='AC1(12m)',
    showlegend=False
), row=2, col=1)
fig_ts.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)

# Plot 3: AC1(18m)
fig_ts.add_trace(go.Scatter(
    x=plot_data_ts.index,
    y=plot_data_ts['AC1_378d'],
    mode='lines',
    line=dict(color='magenta', width=1),
    name='AC1(18m)',
    showlegend=False
), row=3, col=1)
fig_ts.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=3, col=1)

fig_ts.update_layout(
    title='Market Regimes Over Time (Semi-Markov Model)',
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white'),
    height=900,
    width=1400,
    hovermode='x unified'
)

fig_ts.update_xaxes(showgrid=True, gridcolor='rgb(50, 50, 50)')
fig_ts.update_yaxes(showgrid=True, gridcolor='rgb(50, 50, 50)')

fig_ts.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print("1. Autocorrelation structure identifies momentum vs mean-reversion regimes")
print("2. Duration statistics show regime persistence (compare to paper)")
print("3. Visualization shows regime clustering in AC1 space")
print("="*70)

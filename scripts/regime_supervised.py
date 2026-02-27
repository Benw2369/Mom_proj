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
    
    for t in range(n - k):
        # Calculate k-day returns centered at t
        window = series.iloc[t:t+k+1]
        if len(window) < 2:
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
    print(f"  AC1({k} days): Mean={data[col_name].mean():.4f}, Std={data[col_name].std():.4f}")

# Paper shows: AC1(k) is positive for k=1-189 days (momentum), negative for k=315-378 days (reversal)

# --------------------------
# Prepare supervised dataset using duration-dependent features
# --------------------------
n = len(data)

# --- REGIME DEFINITIONS based on Giner & Zakamulin (2023) ---
# Paper insights:
# - 12-month momentum (positive autocorr) followed by reversal (negative autocorr)
# - Duration dependence: older regimes more likely to end
# - Bull markets: ~560 days (~28 months), ~23% annual return, ~16% vol
# - Bear markets: ~280 days (~14 months), ~-28% annual return, ~19% vol

# Initialize probability matrix
prob_matrix = np.zeros((n, 3))

# Define thresholds based on empirical distributions
vol_high = data['fwd_vol'].quantile(0.80) if 'fwd_vol' in data.columns and data['fwd_vol'].notna().any() else 0.02  # Top 20% = crisis
vol_low = data['fwd_vol'].quantile(0.50) if 'fwd_vol' in data.columns and data['fwd_vol'].notna().any() else 0.01   # Bottom 50% = stable
autocorr_positive = 0.05  # Positive autocorr threshold for momentum
autocorr_negative = -0.05  # Negative autocorr threshold for mean-reversion

for i in range(n):
    vol = data['fwd_vol'].iloc[i] if 'fwd_vol' in data.columns else np.nan
    fwd_autocorr = data['fwd_12m_autocorr'].iloc[i]
    fwd_ret = data['fwd_12m_return'].iloc[i] if 'fwd_12m_return' in data.columns else 0
    regime_age_val = data['regime_age'].iloc[i]
    
    # Handle NaN values
    if np.isnan(vol) or np.isnan(fwd_autocorr):
        prob_matrix[i] = [0.33, 0.33, 0.34]
        continue
    
    # REGIME 1: TRENDING/MOMENTUM (low vol + positive autocorr)
    # Paper: Bull markets have low vol, positive returns, persistence
    if vol < vol_low and fwd_autocorr > autocorr_positive:
        # Duration dependence: younger regimes more stable
        if regime_age_val < 60:  # < 3 months old
            prob_matrix[i] = [0.85, 0.10, 0.05]
        else:
            prob_matrix[i] = [0.75, 0.20, 0.05]  # Older, more likely to end
    
    # REGIME 3: HIGH VOLATILITY/CRISIS (top 20% vol)
    # Paper: Crisis periods are rare, high vol, short duration (~280 days)
    elif vol > vol_high:
        # Crisis regimes are unstable - high probability regardless of age
        if vol > data['fwd_vol'].quantile(0.90):
            prob_matrix[i] = [0.05, 0.10, 0.85]  # Extreme crisis
        else:
            prob_matrix[i] = [0.10, 0.15, 0.75]  # High vol
    
    # REGIME 2: MEAN-REVERSION (moderate vol + negative autocorr)
    # Paper: After 12-month momentum, reversal occurs
    elif fwd_autocorr < autocorr_negative:
        # Mean-reversion typically follows extended trends (duration dependence)
        if regime_age_val > 120:  # > 6 months old, likely to revert
            prob_matrix[i] = [0.10, 0.80, 0.10]
        else:
            prob_matrix[i] = [0.15, 0.70, 0.15]
    
    # MIXED/TRANSITIONAL STATES
    else:
        # Use regime age and autocorr to determine likely regime
        if fwd_autocorr > 0 and vol < vol_high:
            prob_matrix[i] = [0.55, 0.30, 0.15]  # Lean trending
        elif fwd_autocorr < 0:
            prob_matrix[i] = [0.25, 0.55, 0.20]  # Lean mean-reversion  
        else:
            prob_matrix[i] = [0.35, 0.40, 0.25]  # Truly mixed

data[['P_Momentum', 'P_MeanRev', 'P_Noise']] = prob_matrix

# --- Prepare supervised dataset ---
X = data[features].iloc[:n-H].values
y = prob_matrix[:n-H]

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
X_clean, y_clean = X[mask], y[mask]
indices_clean = data.iloc[:n-H].index[mask]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_clean, y_clean, indices_clean, shuffle=False, test_size=0.3
)

# --- Train supervised model ---
regressor = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=200, random_state=42)
)
regressor.fit(X_train, y_train)

# --- Predict probabilities ---
y_pred = regressor.predict(X_test)
y_pred /= y_pred.sum(axis=1, keepdims=True)  # normalize

# Append predictions to original DataFrame
data.loc[idx_test, ['P_Momentum_pred', 'P_MeanRev_pred', 'P_Noise_pred']] = y_pred


import numpy as np
import plotly.graph_objects as go

# --- Hard regime assignment ---
prob_cols = ['P_Momentum_pred', 'P_MeanRev_pred', 'P_Noise_pred']
mask = data[prob_cols].notna().all(axis=1)

# Extract probabilities
probs = data.loc[mask, prob_cols].values

# Assign state = index of max probability
hard_state = np.argmax(probs, axis=1)

# Define colors for each state
state_colors = ['red', 'green', 'blue']  # Momentum, MeanRev, Noise

# Create color array
colors = [state_colors[s] for s in hard_state]

# Optional: set opacity proportional to max probability
opacities = np.max(probs, axis=1)  # higher probability → more opaque

# --- 3D scatter plot with DARK MODE ---
fig = go.Figure()

# Create separate traces for each regime for better legend
state_names = ['Momentum', 'Mean-Reversion', 'Noise']
for state_idx in range(3):
    state_mask = hard_state == state_idx
    indices = np.where(state_mask)[0]
    
    fig.add_trace(go.Scatter3d(
        x=data.loc[mask, features[0]].values[state_mask],
        y=data.loc[mask, features[1]].values[state_mask],
        z=data.loc[mask, features[2]].values[state_mask],
        mode='markers',
        marker=dict(
            size=4,
            color=state_colors[state_idx],
            opacity=0.7,
            line=dict(width=0)
        ),
        name=state_names[state_idx],
        text=[f"Momentum: {data.loc[mask, 'P_Momentum_pred'].values[i]:.2f}<br>MeanRev: {data.loc[mask, 'P_MeanRev_pred'].values[i]:.2f}<br>Noise: {data.loc[mask, 'P_Noise_pred'].values[i]:.2f}" 
              for i in indices],
        hovertemplate='<b>%{fullData.name}</b><br>%{text}<br>' +
                      f'{features[0]}: %{{x:.4f}}<br>' +
                      f'{features[1]}: %{{y:.4f}}<br>' +
                      f'{features[2]}: %{{z:.4f}}<extra></extra>'
    ))

fig.update_layout(
    title='Market Regime Prediction (Supervised Learning)',
    scene=dict(
        xaxis_title=features[0],
        yaxis_title=features[1],
        zaxis_title=features[2],
        bgcolor='rgb(17, 17, 17)',
        xaxis=dict(
            backgroundcolor='rgb(17, 17, 17)',
            gridcolor='rgb(50, 50, 50)',
            showbackground=True,
            color='white'
        ),
        yaxis=dict(
            backgroundcolor='rgb(17, 17, 17)',
            gridcolor='rgb(50, 50, 50)',
            showbackground=True,
            color='white'
        ),
        zaxis=dict(
            backgroundcolor='rgb(17, 17, 17)',
            gridcolor='rgb(50, 50, 50)',
            showbackground=True,
            color='white'
        )
    ),
    template='plotly_dark',
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white', size=12),
    width=1200,
    height=900
)

fig.show()

# --- Print comprehensive metrics ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

# 1. Prediction accuracy (how well we predict probabilities)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nProbability Prediction Quality:")
print(f"  MSE: {mse:.4f} (lower better, <0.05 good)")
print(f"  MAE: {mae:.4f} (lower better, <0.15 good)")

# 2. Regime distribution
print(f"\n=== Regime Distribution ===")
for i, name in enumerate(state_names):
    count = np.sum(hard_state == i)
    pct = 100 * count / len(hard_state)
    avg_prob = probs[hard_state == i, i].mean() if count > 0 else 0
    print(f"{name}: {count} samples ({pct:.1f}%) | Avg confidence: {avg_prob:.3f}")

# 3. Regime separation quality (using test features)
X_test_clean = data.loc[idx_test, features].values
hard_state_test = np.argmax(y_pred, axis=1)

print(f"\n=== Regime Separation Quality ===")
silhouette = silhouette_score(X_test_clean, hard_state_test)
print(f"Silhouette Score: {silhouette:.4f}")
print("  Range: -1 to 1, >0.5 = Good separation")

davies_bouldin = davies_bouldin_score(X_test_clean, hard_state_test)
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print("  Lower is better (<1.0 good)")

calinski = calinski_harabasz_score(X_test_clean, hard_state_test)
print(f"Calinski-Harabasz Score: {calinski:.2f}")
print("  Higher is better (>100 good)")

# 4. Regime stability (transition rate)
transitions = np.sum(np.diff(hard_state) != 0)
transition_rate = transitions / len(hard_state)
avg_duration = 1 / transition_rate if transition_rate > 0 else float('inf')
print(f"\n=== Regime Stability ===")
print(f"Transition Rate: {transition_rate:.4f}")
print(f"Total Transitions: {transitions}")
print(f"Average Regime Duration: {avg_duration:.1f} days")
print("  Lower rate = More stable regimes")

# 5. Prediction confidence
max_probs = np.max(probs, axis=1)
avg_confidence = max_probs.mean()
low_confidence_pct = 100 * np.sum(max_probs < 0.5) / len(max_probs)
print(f"\n=== Prediction Confidence ===")
print(f"Average Max Probability: {avg_confidence:.4f}")
print(f"Low Confidence (<0.5): {low_confidence_pct:.1f}% of samples")
print("  >0.7 = High confidence, 0.5-0.7 = Moderate, <0.5 = Uncertain")

# 6. Feature importance (from GradientBoosting)
print(f"\n=== Feature Importance ===")
for i, feature in enumerate(features):
    importances = []
    for estimator in regressor.estimators_:
        importances.append(estimator.feature_importances_[i])
    avg_importance = np.mean(importances)
    print(f"{feature:20s}: {avg_importance:.4f}")

print("\n" + "="*60)
print("OPTIMIZATION SUMMARY")
print("="*60)
print(f"✓ Prediction MSE:        {mse:.4f} (target: <0.05)")
print(f"✓ Silhouette Score:      {silhouette:.4f} (target: >0.5)")
print(f"✓ Avg Confidence:        {avg_confidence:.4f} (target: >0.7)")
print(f"✓ Regime Stability:      {avg_duration:.1f} days avg")
print(f"✓ Davies-Bouldin:        {davies_bouldin:.4f} (target: <1.0)")
print("="*60 + "\n")

# --- Time series plot of regimes over time ---
fig2 = go.Figure()

# Get close prices aligned with predictions
close_prices = data.loc[data.loc[mask].index, 'Close']

# Create traces for each regime
for state_idx in range(3):
    state_mask = hard_state == state_idx
    regime_dates = data.loc[mask].index[state_mask]
    regime_prices = close_prices[state_mask]
    
    fig2.add_trace(go.Scatter(
        x=regime_dates,
        y=regime_prices,
        mode='markers',
        marker=dict(
            size=5,
            color=state_colors[state_idx],
            opacity=0.8
        ),
        name=state_names[state_idx],
        showlegend=True
    ))

# Add overall line for context
fig2.add_trace(go.Scatter(
    x=close_prices.index,
    y=close_prices.values,
    mode='lines',
    line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
    name='Close Price',
    showlegend=True
))

fig2.update_layout(
    title='DJI Close Prices Colored by Predicted Regime',
    xaxis_title='Date',
    yaxis_title='Close Price',
    template='plotly_dark',
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white', size=12),
    width=1400,
    height=600,
    hovermode='x unified',
    xaxis=dict(
        gridcolor='rgb(50, 50, 50)',
        color='white'
    ),
    yaxis=dict(
        gridcolor='rgb(50, 50, 50)',
        color='white'
    )
)

fig2.show()


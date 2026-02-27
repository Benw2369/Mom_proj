import os
import sys
from hmmlearn import hmm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

model_results = {}
scaler = StandardScaler()

# Add mom_dash_app to Python path
sys.path.insert(0, '/Users/benwilliams/VSC/Mom_proj/mom_dash_app')
from data_handler import sql_dbs
from mom_strat_new import Indicators

# Use Indicators with the DJI index (^DJI is the ticker in the database)
ind = Indicators('DJI', '^DJI')
ind.calc_mom_indicators(name='mom_1,4', weights=[0.25,0.25,0.25,0.25,0,0,0,0,0,0,0,0], periodicity='daily')
data = ind.data

# Feature 1: AR(lag) autocorrelation (rolling)
def calc_rolling_ar(series, window, lag):
    ar_values = []
    for i in range(len(series)):
        if i < window:
            ar_values.append(np.nan)
        else:
            window_data = series.iloc[i-window:i]
            if len(window_data) < lag + 10:
                ar_values.append(np.nan)
                continue
            
            # Calculate autocorrelation at specified lag
            returns = window_data.values
            mean_ret = returns.mean()
            numerator = np.sum((returns[:-lag] - mean_ret) * (returns[lag:] - mean_ret))
            denominator = np.sum((returns - mean_ret) ** 2)
            
            if denominator == 0:
                ar_values.append(np.nan)
            else:
                ar_values.append(numerator / denominator)
    return ar_values

def calculate_separation_metrics(data_features, features, hidden_states, n_states):
    """Calculate separation quality metrics"""
    metrics = {}
    
    for feature in features:
        state_means = []
        state_vars = []
        
        for state in range(n_states):
            mask = hidden_states == state
            state_data = data_features[feature][mask]
            state_means.append(state_data.mean())
            state_vars.append(state_data.var())
        
        between_var = np.var(state_means)
        within_var = np.mean(state_vars)
        separation_ratio = between_var / within_var if within_var > 0 else 0
        
        overlaps = []
        for i in range(n_states):
            for j in range(i+1, n_states):
                mean_i, std_i = state_means[i], np.sqrt(state_vars[i])
                mean_j, std_j = state_means[j], np.sqrt(state_vars[j])
                distance = abs(mean_i - mean_j)
                avg_std = (std_i + std_j) / 2
                separation = distance / avg_std if avg_std > 0 else 0
                overlaps.append(separation)
        
        avg_separation = np.mean(overlaps)
        
        metrics[feature] = {
            'separation_ratio': separation_ratio,
            'avg_separation_distance': avg_separation,
        }
    
    return metrics

# Calculate regime features
# Test different windows and lags to optimize separation
OPTIMIZE_PARAMS = True  # Set to False to skip optimization
n_states = 3  # Define number of states early

if OPTIMIZE_PARAMS:
    window_options = [30, 60, 90, 120]
    lag_options = [5, 10, 15, 20]
    
    best_score = 0
    best_params = None
    results = []
    
    print("Testing parameter combinations...")
    for window in window_options:
        for lag in lag_options:
            # Skip if lag is too large for window
            if lag + 20 > window:
                continue
            
            # Compute features with current params
            temp_data = data.copy()
            
            temp_data[f'AR{lag}_autocorr'] = calc_rolling_ar(temp_data['daily_ret'], window, lag)
            temp_data['cum_return'] = temp_data['daily_ret'].rolling(window).sum()
            temp_data['rolling_vol'] = temp_data['daily_ret'].rolling(window).std()
            temp_data['vol_slope'] = temp_data['rolling_vol'].diff(5)
            
            features = [f'AR{lag}_autocorr', 'cum_return', 'vol_slope']
            temp_features = temp_data[features].copy().dropna()
            
            if len(temp_features) < 100:
                continue
            
            # Fit HMM
            X_temp = scaler.fit_transform(temp_features[features].values)
            temp_model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', 
                                         n_iter=1000, random_state=42, tol=1e-4)
            temp_model.fit(X_temp)
            temp_states = temp_model.predict(X_temp)
            
            # Calculate separation
            temp_metrics = calculate_separation_metrics(temp_features, features, temp_states, n_states)
            score = np.mean([m['separation_ratio'] for m in temp_metrics.values()])
            
            results.append({
                'window': window,
                'lag': lag,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {'window': window, 'lag': lag}
            
            print(f"  Window={window:3d}, Lag={lag:2d} → Score: {score:.4f}")
    
    print(f"\n🏆 Best parameters: Window={best_params['window']}, Lag={best_params['lag']}, Score={best_score:.4f}\n")
    window = best_params['window']
    lag = best_params['lag']
else:
    window = 60  # Rolling window for calculations
    lag = 5      # Primary lag for autocorrelation

# Use a larger window to ensure all AR lags fit
# Need window > max(lag) + 20 for reliable calculation
window = max(window, 90)  # Ensure at least 90 days for AR30

# Now calculate multiple AR features with different lags
# This captures autocorrelation structure at different timescales
lag1 = 5   # Short-term (1 week)
lag2 = 15  # Medium-term (3 weeks)
lag3 = 30  # Long-term (6 weeks)

print(f"Computing AR features with window={window}, lags={[lag1, lag2, lag3]}")

data[f'AR{lag1}_autocorr'] = calc_rolling_ar(data['daily_ret'], window, lag1)
data[f'AR{lag2}_autocorr'] = calc_rolling_ar(data['daily_ret'], window, lag2)
data[f'AR{lag3}_autocorr'] = calc_rolling_ar(data['daily_ret'], window, lag3)

# Select 3 AR features at different lags for maximum separation
# AR5 captures short-term momentum vs MR patterns (1 week)
# AR15 captures medium-term autocorrelation (3 weeks)
# AR30 captures long-term autocorrelation (6 weeks)
features = [f'AR{lag1}_autocorr', f'AR{lag2}_autocorr', f'AR{lag3}_autocorr']
n_states = 3

# Select only the features we need
data_features = data[features].copy()
data_features = data_features.dropna()

print(f"\nFeature data shape: {data_features.shape}")
print(f"Features: {features}")

if len(data_features) < 100:
    raise ValueError(f"Not enough data after dropna: {len(data_features)} samples")

# NORMALIZE/STANDARDIZE features for better separation
# This is critical - HMM works better when features are on similar scales
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(data_features[features].values)

# Use diagonal covariance for more distinct clusters
# Try different random states or increase iterations for stability
model = hmm.GaussianHMM(
    n_components=n_states, 
    covariance_type='diag',  # Changed from 'full' - forces more separation
    n_iter=2000,              # Increased iterations
    random_state=42,
    tol=1e-4                  # Tighter convergence
)
model.fit(X)

hidden_states = model.predict(X)
data_features['State'] = hidden_states

probs = model.predict_proba(X)
data_features[['P_State0','P_State1','P_State2']] = probs

# Print state emission means to identify regimes
state_means = model.means_

print("\n=== State Emission Means ===")
for i in range(n_states):
    print(f"\nState {i}:")
    print(f"  AR{lag1} autocorr: {state_means[i, 0]:.4f}")
    print(f"  AR{lag2} autocorr: {state_means[i, 1]:.4f}")
    print(f"  AR{lag3} autocorr: {state_means[i, 2]:.4f}")
    print(f"  Count: {np.sum(hidden_states == i)}")

# Smart labeling: assign each state a unique label based on its characteristics
ar1_values = state_means[:, 0]
vol_values = state_means[:, 2]

# Find state with highest AR1 → Momentum
momentum_state = np.argmax(ar1_values)

# Find state with lowest AR1 → Mean-Reversion
mr_state = np.argmin(ar1_values)

# Remaining state → Noise/Sideways
noise_state = [i for i in range(n_states) if i not in [momentum_state, mr_state]][0]

state_labels = {
    momentum_state: 'Momentum',
    mr_state: 'Mean-Reversion',
    noise_state: 'Noise/Sideways'
}

print("\n=== State Labels ===")
for i in range(n_states):
    print(f"State {i} → {state_labels[i]}")

# Map state numbers to labels
data_features['State_Label'] = data_features['State'].map(state_labels)

# Calculate separation metrics for each feature
def calculate_separation_metrics(data_features, features, hidden_states, n_states):
    """
    Calculate how well features separate the states using:
    1. Between-state variance / Within-state variance (higher is better)
    2. Overlap percentage between distributions
    """
    metrics = {}
    
    for feature in features:
        # Calculate means and variances for each state
        state_means = []
        state_vars = []
        
        for state in range(n_states):
            mask = hidden_states == state
            state_data = data_features[feature][mask]
            state_means.append(state_data.mean())
            state_vars.append(state_data.var())
        
        # Between-state variance
        overall_mean = data_features[feature].mean()
        between_var = np.var(state_means)
        
        # Within-state variance (average)
        within_var = np.mean(state_vars)
        
        # Separation ratio (F-statistic like measure)
        separation_ratio = between_var / within_var if within_var > 0 else 0
        
        # Calculate overlap using standard deviations
        overlaps = []
        for i in range(n_states):
            for j in range(i+1, n_states):
                mean_i, std_i = state_means[i], np.sqrt(state_vars[i])
                mean_j, std_j = state_means[j], np.sqrt(state_vars[j])
                
                # Distance between means in units of average std
                distance = abs(mean_i - mean_j)
                avg_std = (std_i + std_j) / 2
                separation = distance / avg_std if avg_std > 0 else 0
                overlaps.append(separation)
        
        avg_separation = np.mean(overlaps)
        
        metrics[feature] = {
            'separation_ratio': separation_ratio,
            'avg_separation_distance': avg_separation,
            'between_var': between_var,
            'within_var': within_var
        }
    
    return metrics

separation_metrics = calculate_separation_metrics(data_features, features, hidden_states, n_states)

print("\n=== Feature Separation Metrics ===")
print("Higher is better - indicates better regime separation\n")
for feature, metrics in separation_metrics.items():
    print(f"{feature}:")
    print(f"  Separation Ratio (Between/Within var): {metrics['separation_ratio']:.4f}")
    print(f"  Avg Distance (in std units):           {metrics['avg_separation_distance']:.4f}")
    print()

# Overall quality score
overall_score = np.mean([m['separation_ratio'] for m in separation_metrics.values()])
print(f"Overall Feature Set Quality: {overall_score:.4f}")
print(f"Window={window}, Lag={lag}\n")

# Additional regime distinctiveness measures
print("=== Regime Distinctiveness Measures ===\n")

# 1. Silhouette Score (measures how well-separated clusters are)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(data_features[features], hidden_states)
print(f"Silhouette Score: {silhouette:.4f}")
print("  Range: -1 to 1, higher is better")
print("  > 0.5 = Good separation, 0.2-0.5 = Moderate, < 0.2 = Poor\n")

# 2. Davies-Bouldin Index (lower is better)
davies_bouldin = davies_bouldin_score(data_features[features], hidden_states)
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print("  Lower is better (0 = perfect separation)\n")

# 3. Calinski-Harabasz Score (variance ratio criterion - higher is better)
calinski = calinski_harabasz_score(data_features[features], hidden_states)
print(f"Calinski-Harabasz Score: {calinski:.2f}")
print("  Higher is better (measures cluster density and separation)\n")

# 4. State transition stability (how often regimes switch)
transitions = np.sum(np.diff(hidden_states) != 0)
transition_rate = transitions / len(hidden_states)
print(f"Regime Transition Rate: {transition_rate:.4f}")
print(f"  Total transitions: {transitions}")
print(f"  Average regime duration: {1/transition_rate:.1f} days" if transition_rate > 0 else "  No transitions")
print("  Lower rate = More stable regimes\n")

# 5. State probability confidence (how confident is the model?)
state_confidence = np.mean(np.max(probs, axis=1))
print(f"Average State Confidence: {state_confidence:.4f}")
print("  Range: 0.33 to 1.0, higher = model is more certain about regime assignments\n")

# 6. State balance (are states evenly distributed?)
state_counts = [np.sum(hidden_states == i) for i in range(n_states)]
state_balance = np.std(state_counts) / np.mean(state_counts)
print(f"State Balance (CV): {state_balance:.4f}")
print(f"  State counts: {state_counts}")
print("  Lower = More balanced distribution, Higher = Some states dominate\n")

# Summary score
print("=" * 50)
print("OVERALL REGIME QUALITY SUMMARY")
print("=" * 50)
print(f"✓ Feature Separation:     {overall_score:.4f} (higher better)")
print(f"✓ Silhouette Score:       {silhouette:.4f} (>0.5 good)")
print(f"✓ Davies-Bouldin:         {davies_bouldin:.4f} (lower better)")
print(f"✓ State Confidence:       {state_confidence:.4f} (>0.7 good)")
print(f"✓ Regime Stability:       {1/transition_rate:.1f} days avg duration")
print("=" * 50 + "\n")


# Create interactive 3D visualization with Plotly
colors = ['red', 'blue', 'green']

# Create traces for each state
traces = []
for i in range(n_states):
    mask = hidden_states == i
    trace = go.Scatter3d(
        x=data_features[features[0]][mask],
        y=data_features[features[1]][mask],
        z=data_features[features[2]][mask],
        mode='markers',
        marker=dict(
            size=3,
            color=colors[i],
            opacity=0.6
        ),
        name=f"State {i}: {state_labels[i]}",
        text=[f'Date: {idx}<br>State: {state_labels[i]}' for idx in data_features.index[mask]],
        hovertemplate='<b>%{text}</b><br>' +
                      f'{features[0]}: %{{x:.4f}}<br>' +
                      f'{features[1]}: %{{y:.4f}}<br>' +
                      f'{features[2]}: %{{z:.4f}}<br>' +
                      '<extra></extra>'
    )
    traces.append(trace)

# Create figure
fig = go.Figure(data=traces)

fig.update_layout(
    title='Market Regimes: Momentum vs Mean-Reversion vs Noise',
    scene=dict(
        xaxis_title=features[0],
        yaxis_title=features[1],
        zaxis_title=features[2],
        bgcolor='rgb(17, 17, 17)',
        xaxis=dict(
            backgroundcolor='rgb(17, 17, 17)',
            gridcolor='rgb(50, 50, 50)',
            showbackground=True
        ),
        yaxis=dict(
            backgroundcolor='rgb(17, 17, 17)',
            gridcolor='rgb(50, 50, 50)',
            showbackground=True
        ),
        zaxis=dict(
            backgroundcolor='rgb(17, 17, 17)',
            gridcolor='rgb(50, 50, 50)',
            showbackground=True
        )
    ),
    width=1000,
    height=800,
    template='plotly_dark',
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white')
)

fig.show()

# Create time series plot of Close prices with regime coloring
fig2 = go.Figure()

# Get close prices aligned with data_features
close_prices = data.loc[data_features.index, 'Close']

# Create traces for each regime
for i in range(n_states):
    mask = hidden_states == i
    regime_dates = data_features.index[mask]
    regime_prices = close_prices[mask]
    
    fig2.add_trace(go.Scatter(
        x=regime_dates,
        y=regime_prices,
        mode='markers',
        marker=dict(
            size=4,
            color=colors[i],
            opacity=0.7
        ),
        name=f"{state_labels[i]}",
        showlegend=True
    ))

# Add overall line for context
fig2.add_trace(go.Scatter(
    x=close_prices.index,
    y=close_prices.values,
    mode='lines',
    line=dict(color='gray', width=1),
    name='Close Price',
    opacity=0.3,
    showlegend=True
))

fig2.update_layout(
    title='DJI Close Prices Colored by Market Regime',
    xaxis_title='Date',
    yaxis_title='Close Price',
    template='plotly_dark',
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white'),
    width=1200,
    height=600,
    hovermode='x unified'
)

fig2.show()

# Create distribution plots for each feature by state
from plotly.subplots import make_subplots

fig3 = make_subplots(
    rows=len(features), 
    cols=1,
    subplot_titles=[f'Distribution of {feat}' for feat in features],
    vertical_spacing=0.08
)

colors = ['red', 'blue', 'green']

for feat_idx, feature in enumerate(features):
    for state_idx in range(n_states):
        mask = hidden_states == state_idx
        feature_data = data_features[feature][mask]
        
        fig3.add_trace(
            go.Histogram(
                x=feature_data,
                name=f"{state_labels[state_idx]}",
                marker_color=colors[state_idx],
                opacity=0.6,
                nbinsx=50,
                legendgroup=f"state{state_idx}",
                showlegend=(feat_idx == 0)  # Only show legend for first subplot
            ),
            row=feat_idx + 1,
            col=1
        )

fig3.update_layout(
    title='Feature Distributions by Market Regime',
    template='plotly_dark',
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white'),
    height=300 * len(features),
    width=1000,
    barmode='overlay'
)

fig3.update_xaxes(title_text='Value')
fig3.update_yaxes(title_text='Frequency')

fig3.show()

# X is your feature matrix
log_likelihood = model.score(X)
state_distinctiveness = np.mean(np.max(probs, axis=1))

print("Log-likelihood of the fitted model:", log_likelihood)
print("Average maximum state probability (distinctiveness):", state_distinctiveness)


model_results[tuple(features)] = {
    'log_likelihood': model.score(X),
    # 'transmat': model.transmat_,
    # 'means': model.means_,
    'state_distinctiveness': state_distinctiveness
}














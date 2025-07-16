import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
from regime_detection import MacroPCA
from regime_detection import RegimeHMM
from simulate_strategy import simulate_strategy
from simulate_strategy import simulate_sharpe_weighted_strategy
from simulate_strategy import simulate_vol_scaled_strategy
from simulate_strategy import cumulative_comparison_plot
from simulate_strategy import simulate_filtered_strategy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# Load the FF5 dataset
ff5_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'
fama_french_5 = pd.read_csv(ff5_url, compression='zip', skiprows=2)

fama_french_5 = fama_french_5.rename(columns={'Unnamed: 0': 'Date'})
fama_french_5 = fama_french_5[fama_french_5['Date'].str.strip().str.len() == 6]
fama_french_5['Date'] = pd.to_datetime(fama_french_5['Date'], format="%Y%m") + pd.offsets.MonthEnd(0)

# Retreive the momentum data

momentum_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
momentum = pd.read_csv(momentum_url, compression='zip', skiprows=12)
momentum = momentum.rename(columns={'Unnamed: 0': 'Date'})
momentum = momentum[momentum['Date'].str.strip().str.len() == 6]
momentum['Date'] = pd.to_datetime(momentum['Date'], format="%Y%m") + pd.offsets.MonthEnd(0)

# Merge the datasets on Date to get all factors
all_factors = pd.merge(fama_french_5, momentum, how="inner", on='Date')  # Merge the momentum to the other factors on date
all_factors.set_index('Date', inplace=True)

all_factors = all_factors.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric

# Get the macroeconomic data
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
f = Fred(api_key=fred_api_key)  # Instantiate FRED with API key

# Define series of interest
series = {
    'VIX': 'VIXCLS',
    '10Y': 'GS10',
    '2Y': 'GS2',
    'YieldCurve': 'T10Y2Y',
    'BAA': 'BAA',  # Corporate bond yield (credit risk)
    'CPI': 'CPIAUCSL',
    'UnemploymentRate': 'UNRATE',
    'FedFundsRate': 'FEDFUNDS',
    'GDP_YoY': 'A191RL1Q225SBEA'
}

# Set the start and end dates
start_date = '1963-07-01'
end_date = '2025-05-31'

# Fetch the data from FRED
macro_data = {}
for name, code in series.items():
    macro_data[name] = f.get_series(code, observation_start=start_date, observation_end=end_date)

macro_df = pd.DataFrame(macro_data)  # Convert to a dataframe
macro_df['Date'] = macro_df.index  # Add date as a column

# Compute derived columns
macro_df['CreditSpread'] = macro_df['BAA'] - macro_df['10Y']
macro_df['Inflation_YoY'] = macro_df['CPI'].pct_change(periods=12) * 100
macro_df['GDP_YoY'] = macro_df['GDP_YoY'].ffill()

# Reset index and reorder
macro_df.reset_index(drop=True, inplace=True)
macro_df = macro_df[['Date', 'VIX', '2Y', '10Y', 'YieldCurve', 'CreditSpread',
                     'FedFundsRate', 'Inflation_YoY', 'UnemploymentRate', 'GDP_YoY']]

# Resample to monthly in line with factor data
macro_df.set_index('Date', inplace=True)
macro_df = macro_df.resample('MS').mean()
macro_df.reset_index(inplace=True)

# Move macro data to date where that all the observations are present
macro_df.set_index('Date', inplace=True)
macro_df.index = pd.to_datetime(macro_df.index)
macro_df = macro_df[macro_df.index >= '1990-01-01']

# Run PCA on macroeconomic data
m_model = MacroPCA(data=macro_df, n_components=5)
m_model.standardise()
pc_df_m = m_model.run_pca()

# Plot the principal components over time
pc_df_m.index = pd.to_datetime(pc_df_m.index)

plt.figure(figsize=(14, 8))
for i in range(5):
    plt.plot(pc_df_m.index, pc_df_m.iloc[:, i], label=f'PC{i+1}')
plt.title('Principal Components Over Time')
plt.xlabel('Date')
plt.ylabel('Component Value')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))  # Every 2 years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  # Rotate x-axis labels for readability

plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('pca_timeline.png', dpi=300)  # Save the plot

# Fit HMM to monthly PCA output
m_hmm = RegimeHMM(pca_output=pc_df_m, n_regimes=4, covariance_type='full')
m_hmm.fit()
m_hmm.plot_pc_with_regimes("Monthly PC with HMM Regimes")
m_hmm.get_transition_matrix()

# Attach the regime labels to the PCA output and dates
monthly_regimes = m_hmm.pca_output[['PC1', 'PC2', 'PC3', 'PC4', 'Regime']].copy()

# This next section will be about factor performance based on the regimes detected

monthly_regimes['LaggedRegime'] = monthly_regimes['Regime'].shift(1)

all_factors.index = pd.to_datetime(all_factors.index)
merged = all_factors.merge(monthly_regimes, left_index=True, right_index=True, how='inner')
merged.dropna(subset=['LaggedRegime'], inplace=True)
merged['LaggedRegime'] = merged['LaggedRegime'].astype(int)
merged.to_csv('merged_factors_with_regimes.csv')

# Define factor columns you want to evaluate
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
# Group by LaggedRegime to evaluate factor performance after regime is known

grouped = merged.groupby('LaggedRegime')[factor_cols]
# Compute mean and standard deviation of factor returns in each regime
performance = grouped.agg(['mean', 'std'])

# Calculate Sharpe ratio for each factor in each regime (mean / std)
sharpe_ratios = performance.xs('mean', axis=1, level=1) / performance.xs('std', axis=1, level=1)
sharpe_ratios.columns = [f'{col}_Sharpe' for col in sharpe_ratios.columns]

# Flatten the multi-index columns of performance DataFrame
performance.columns = ['_'.join(col) for col in performance.columns]

# Combine mean, std, and Sharpe into one final table
performance_summary = pd.concat([performance, sharpe_ratios], axis=1)

# Extract only Sharpe columns
sharpe_df = performance_summary[[col for col in performance_summary.columns if col.endswith('_Sharpe')]]

# Clean column names
sharpe_df.columns = [col.replace('_Sharpe', '') for col in sharpe_df.columns]

# Only keep positive Sharpe ratios
sharpe_pos_df = sharpe_df.copy()
sharpe_pos_df[sharpe_pos_df < 0] = 0  # set negative Sharpe values to 0

# Normalize Sharpe ratios within each regime to get weights
sharpe_weights = sharpe_pos_df.div(sharpe_pos_df.sum(axis=1), axis=0)

# Drop regimes where all factors had 0 Sharpe
sharpe_weights = sharpe_weights.dropna(how='all')

# Define how many top factors to select per regime
TOP_N = 2

# Get top-N factors per regime based on Sharpe ratio
regime_factors = {}
for regime in sharpe_df.index:
    ranked = sharpe_df.loc[regime].sort_values(ascending=False)
    top_factors = ranked[ranked > 0].head(TOP_N).index.tolist()
    regime_factors[regime] = top_factors

# Display result
print("=== Regime-Based Factor Allocation Strategy ===")
for regime, factors in regime_factors.items():
    print(f"Regime {regime}: {', '.join(factors)}")

# Transpose so we can plot regimes on x-axis
sharpe_df_T = sharpe_df.T

# Plot
sharpe_df_T.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Sharpe Ratios by Factor and Regime')
plt.xlabel('Factor')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("sharpe_ratios_by_regime.png", dpi=300)

# Simulate the strategy
strategy_returns = simulate_strategy(merged, regime_factors)
strategy_weighted_returns = simulate_sharpe_weighted_strategy(merged, sharpe_weights)
strategy_filtered_returns = simulate_filtered_strategy(merged, regime_factors, pc_filter_col='PC1', pc_filter_threshold=0, vix_filter=True, vix_threshold=20)
strategy_excess = strategy_filtered_returns - merged.loc[strategy_filtered_returns.index, 'RF']

# Build list of all factors used across regimes
factors_used = sorted(set(f for sublist in regime_factors.values() for f in sublist))
# Compute 12-month rolling std for each factor
rolling_vol = merged[factors_used].rolling(window=12).std()
strategy_vol_scaled_returns = simulate_vol_scaled_strategy(merged, regime_factors, rolling_vol)

# Benchmark comparison

# Get benchmark returns (e.g., Mkt-RF)
benchmark_returns = merged['Mkt-RF']

# Cumulative return plots

cumulative_comparison_plot(strategy_returns, benchmark_returns, "PCA=5_N=2_Equal_Weight_cumulative_returns_strategy_vs_benchmark")
cumulative_comparison_plot(strategy_weighted_returns, benchmark_returns, "PCA=5_N=2_Sharpe_Weighted_cumulative_returns_strategy_vs_benchmark")
cumulative_comparison_plot(strategy_excess, benchmark_returns, "PCA=5_N=2_Filtered_cumulative_returns_strategy_vs_benchmark") # Seems to be the same as equal weight
cumulative_comparison_plot(strategy_vol_scaled_returns, benchmark_returns, "PCA=5_N=2_Vol_Scaled_cumulative_returns_strategy_vs_benchmark")

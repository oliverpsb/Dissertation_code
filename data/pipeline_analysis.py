import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
from regime_detection import MacroPCA
from regime_detection import RegimeHMM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import logging
from scipy.stats import f_oneway

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
all_factors.to_csv('analysis_output/factors_data.csv', index=True)
logging.info('Created and saved factors dataframe')

print()

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
macro_df.to_csv('analysis_output/macro_data.csv', index=True)

logging.info('Fetched, processed and saved macroeconomic data as macro_data.csv')

logging.info('Running PCA on macroeconomic data...')
# Run PCA on macroeconomic data
n_comps = 5  # Number of components to extract
m_model = MacroPCA(data=macro_df, n_components=n_comps)
m_model.standardise()
pc_df_m = m_model.run_pca()
scree_plot = m_model.plot_scree(save_path=f'analysis_output/pc{n_comps}_scree_plot.png')

logging.info('PCA justification steps...')
explained_var = m_model.get_explained_variance()
explained_series = pd.Series(explained_var, index=[f'PC{i+1}' for i in range(len(explained_var))])
explained_series.to_csv(f'analysis_output/pc{n_comps}_explained_variance.csv', index=True)


pc_df_m.to_csv(f'analysis_output/pc{n_comps}_df_m.csv', index=True)  # Save PCA output

logging.info(f'PCA completed. Saved PCA output to pc{n_comps}_df_m.csv')

logging.info('Plotting principal components over time...')
# Plot the principal components over time
pc_df_m.index = pd.to_datetime(pc_df_m.index)

plt.figure(figsize=(14, 8))
for i in range(n_comps):
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
plt.savefig(f'analysis_output/pca{n_comps}_timeline.png', dpi=300)  # Save the plot

logging.info(f'PCA timeline plot saved as pca{n_comps}_timeline.png')

logging.info('Fitting HMM to monthly PCA output...')
# Fit HMM to monthly PCA output
num_reg = 3
m_hmm = RegimeHMM(pca_output=pc_df_m, n_regimes=num_reg, covariance_type='full', simulate=False)
m_hmm.fit()

logging.info('Running BIC comparison for different regime counts...')
m_hmm.compare_bic(min_regimes=2, max_regimes=8)
logging.info('BIC comparison completed and saved to analysis_output/hmm_bic_comparison.csv')

m_hmm.plot_pc_with_regimes("Monthly PC with HMM Regimes", n_pca_components=n_comps)
logging.info(f'HMM fitted and plot saved as pc{n_comps}_with_regimes.png.')
transition_matrix = m_hmm.get_transition_matrix()
transition_matrix_df = pd.DataFrame(transition_matrix)
transition_matrix_df.to_csv(f'analysis_output/pc{n_comps}_r{num_reg}_transition_matrix.csv')
logging.info(f'Transition matrix obtained from HMM model and saved as pc{n_comps}_r{num_reg}_transition_matrix.csv')

# Attach the regime labels to the PCA output and dates
monthly_regimes = m_hmm.pca_output[['PC1', 'PC2', 'PC3', 'PC4', 'Regime']].copy()

# This next section will be about factor performance based on the regimes detected

logging.info('Merging macro PCA with factor data...')

monthly_regimes['LaggedRegime'] = monthly_regimes['Regime'].shift(1)

all_factors.index = pd.to_datetime(all_factors.index)
merged = all_factors.merge(monthly_regimes, left_index=True, right_index=True, how='inner')
merged.dropna(subset=['LaggedRegime'], inplace=True)
merged['LaggedRegime'] = merged['LaggedRegime'].astype(int)
merged.to_csv(f'analysis_output/pc{n_comps}_r{num_reg}_merged_factors_with_regimes.csv')

logging.info(f'Merged factors with regimes and saved as pc{n_comps}_r{num_reg}_merged_factors_with_regimes.csv')

logging.info('Evaluating factor performance based on regimes...')
# Define factor columns you want to evaluate
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
# Group by LaggedRegime to evaluate factor performance after regime is known

grouped = merged.groupby('LaggedRegime')[factor_cols]
# Compute mean and standard deviation of factor returns in each regime
performance = grouped.agg(['mean', 'std'])

# Calculate Sharpe ratio for each factor in each regime (mean / std)
sharpe_ratios = performance.xs('mean', axis=1, level=1) / performance.xs('std', axis=1, level=1)
sharpe_ratios.columns = [f'{col}_Sharpe' for col in sharpe_ratios.columns]

# Calculate Hit Ratio
hit_ratios = merged.groupby('LaggedRegime')[factor_cols].apply(lambda x: (x > 0).mean())
hit_ratios.columns = [f'{col}_HitRatio' for col in hit_ratios.columns]

# Calculate Sortino Ratio
def sortino_ratio(x):
    downside = x[x < 0]
    return x.mean() / downside.std() if downside.std !=0 else np.nan

sortino_ratios = merged.groupby('LaggedRegime')[factor_cols].agg(sortino_ratio)
sortino_ratios.columns = [f'{col}_Sortino' for col in sortino_ratios.columns]

# Flatten the multi-index columns of performance DataFrame
performance.columns = ['_'.join(col) for col in performance.columns]

# Combine mean, std, and Sharpe into one final table
performance_summary = pd.concat([performance, sharpe_ratios], axis=1)
performance_summary = pd.concat([performance_summary, hit_ratios, sortino_ratios], axis=1)
performance_summary.to_csv(f'analysis_output/pc{n_comps}_r{num_reg}_factor_performance_summary.csv')

logging.info(f'Factor performance summary saved as pc{n_comps}_r{num_reg}_factor_performance_summary.csv')




logging.info('Implementing ANOVA to compare factor performance across regimes...')
anova_results = {}
for factor in factor_cols:
    samples = [group[factor].values for _, group in merged.groupby('LaggedRegime')]
    f_stat, p_value = f_oneway(*samples)
    anova_results[factor] = {'F-stat': f_stat, 'p-value': p_value}

anova_df = pd.DataFrame(anova_results).T
anova_df.to_csv(f'analysis_output/pc{n_comps}_anova_results.csv')
logging.info(f'ANOVA results saved as pc{n_comps}_anova_results.csv')


logging.info('Calculating full-sample Sharpe ratio for baseline comparison...')
# Calculate full-sample Sharpe ratio for baseline comparison
full_sample_sharpe = all_factors[factor_cols].mean() / all_factors[factor_cols].std()
full_sample_sharpe.name = 'FullSample_Sharpe'
full_sample_sharpe.to_csv(f'analysis_output/pc{n_comps}_full_sample_sharpe.csv')

# Extract Sharpe ratios directly from performance_summary for plotting
sharpe_cols = [col for col in performance_summary.columns if col.endswith('_Sharpe')]
sharpe_df = performance_summary[sharpe_cols].copy()
sharpe_df.columns = [col.replace('_Sharpe', '') for col in sharpe_df.columns]

# === 1. Plot REGIME Sharpe ONLY ===
sharpe_df_T_regimes = sharpe_df.T  # Only regime-wise
sharpe_df_T_regimes.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Sharpe Ratios by Factor and Regime')
plt.xlabel('Factor')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"analysis_output/pc{n_comps}_r{num_reg}_sharpe_ratios_by_regime.png", dpi=300)
logging.info(f'Sharpe ratios by regime plot saved as pc{n_comps}_r{num_reg}_sharpe_ratios_by_regime.png')

# === 2. Add FULL SAMPLE Sharpe and plot comparison ===
sharpe_df_T_with_baseline = sharpe_df_T_regimes.copy()
sharpe_df_T_with_baseline['Full Sample'] = full_sample_sharpe
sharpe_df_T_with_baseline.to_csv(f'analysis_output/pc{n_comps}_r{num_reg}_sharpe_ratios_comparison.csv')

# Reorder to keep 'Full Sample' last
ordered_cols = [col for col in sharpe_df_T_with_baseline.columns if col != 'Full Sample'] + ['Full Sample']
sharpe_df_T_with_baseline = sharpe_df_T_with_baseline[ordered_cols]

sharpe_df_T_with_baseline.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Sharpe Ratios by Factor and Regime (Including Full Sample)')
plt.xlabel('Factor')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"analysis_output/pc{n_comps}_r{num_reg}_sharpe_ratios_with_full_sample.png", dpi=300)
logging.info(f'Sharpe ratio comparison with full-sample baseline saved as pc{n_comps}_r{num_reg}_sharpe_ratios_with_full_sample.png')

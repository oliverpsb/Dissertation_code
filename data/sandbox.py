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

# Load daily Fama-French 5 factors + Momentum
ff5_daily_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
momentum_daily_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

# Load daily FF5
ff5_daily = pd.read_csv(ff5_daily_url, compression='zip', skiprows=3)
ff5_daily = ff5_daily.rename(columns={'Unnamed: 0': 'Date'})
ff5_daily = ff5_daily[ff5_daily['Date'].str.match(r'^\d{8}$')]
ff5_daily['Date'] = pd.to_datetime(ff5_daily['Date'], format='%Y%m%d')

# Load daily Momentum
mom_daily = pd.read_csv(momentum_daily_url, compression='zip', skiprows=13)
mom_daily = mom_daily.rename(columns={'Unnamed: 0': 'Date'})
mom_daily = mom_daily[mom_daily['Date'].str.match(r'^\d{8}$')]
mom_daily['Date'] = pd.to_datetime(mom_daily['Date'], format='%Y%m%d')

# Merge the two datasets
daily_factors = pd.merge(ff5_daily, mom_daily, on='Date', how='inner')
daily_factors = daily_factors.set_index('Date')
daily_factors = daily_factors.apply(pd.to_numeric, errors='coerce') / 100  # Convert to decimals

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

# Resample to monthly for regime anlaysis
macro_df.set_index('Date', inplace=True)
macro_df = macro_df.resample('MS').mean()
macro_df.reset_index(inplace=True)

# Move macro data to date where that all the observations are present
macro_df.set_index('Date', inplace=True)
macro_df.index = pd.to_datetime(macro_df.index)
macro_df = macro_df[macro_df.index >= '1990-01-01']


logging.info('Running PCA on macroeconomic data...')
# Run PCA on macroeconomic data
n_comps = 5  # Number of components to extract
m_model = MacroPCA(data=macro_df, n_components=n_comps)
m_model.standardise()
pc_df_m = m_model.run_pca()
scree_plot = m_model.plot_scree(save_path=f'sandbox_output/pc{n_comps}_scree_plot.png', cumulative=True)

logging.info('PCA justification steps...')
explained_var = m_model.get_explained_variance()
cumulative_explained = explained_var[:n_comps].sum()
print(f'Explained variance for {n_comps} components: {cumulative_explained:.3f}')
explained_series = pd.Series(explained_var, index=[f'PC{i+1}' for i in range(len(explained_var))])
explained_series.to_csv(f'sandbox_output/pc{n_comps}_explained_variance.csv', index=True)
loadings_df = m_model.get_loadings()
loadings_df.to_csv(f'sandbox_output/pc{n_comps}_loadings.csv', index=True)

pc_df_m.to_csv(f'sandbox_output/pc{n_comps}_df_m.csv', index=True)  # Save PCA output


logging.info(f'PCA completed. Saved PCA output to pc{n_comps}_df_m.csv')

logging.info('Plotting principal components over time...')
# Plot the principal components over time
pc_df_m.index = pd.to_datetime(pc_df_m.index)

plt.figure(figsize=(20, 8))
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
plt.savefig(f'sandbox_output/pca{n_comps}_timeline.png', dpi=300)  # Save the plot

logging.info(f'PCA timeline plot saved as pca{n_comps}_timeline.png')

logging.info('Fitting HMM to monthly PCA output...')
# Fit HMM to monthly PCA output
num_reg = 4
m_hmm = RegimeHMM(pca_output=pc_df_m, n_regimes=num_reg, covariance_type='full', simulate=False)
m_hmm.fit()

logging.info('Running BIC comparison for different regime counts...')
m_hmm.compare_bic(min_regimes=2, max_regimes=8)
logging.info('BIC comparison completed and saved to sandbox_output/hmm_bic_comparison.csv')

m_hmm.plot_pc_with_regimes_sand("Monthly PC with HMM Regimes", n_pca_components=n_comps)
logging.info(f'HMM fitted and plot saved as pc{n_comps}_with_regimes.png.')
transition_matrix = m_hmm.get_transition_matrix()
transition_matrix_df = pd.DataFrame(transition_matrix)
transition_matrix_df.to_csv(f'sandbox_output/pc{n_comps}_r{num_reg}_transition_matrix.csv')
logging.info(f'Transition matrix obtained from HMM model and saved as pc{n_comps}_r{num_reg}_transition_matrix.csv')

# Attach the regime labels to the PCA output and dates
monthly_regimes = m_hmm.pca_output[['PC1', 'PC2', 'PC3', 'PC4', 'Regime']].copy()

# Forward-fill monthly regime to each day
monthly_regimes_daily = monthly_regimes[['Regime']].copy()
monthly_regimes_daily.index = pd.to_datetime(monthly_regimes_daily.index)
daily_factors.index = pd.to_datetime(daily_factors.index)

# Merge by forward-filling regime onto daily factors
daily_factors_with_regime = pd.merge_asof(
    daily_factors.sort_index(),
    monthly_regimes_daily.sort_index(),
    left_index=True,
    right_index=True,
    direction='backward'
)


# Compite PC stats by regime
pc_stats_by_regime = pc_df_m.groupby('Regime').agg(['mean', 'std'])
pc_stats_by_regime.to_csv(f'sandbox_output/pc{n_comps}_r{num_reg}_pc_summary_by_regime.csv')

# Visualise PC means by regime
pc_means = pc_stats_by_regime.xs('mean', axis=1, level=1)
pc_means.T.plot(kind='bar', figsize=(12, 6))
plt.title('Principal Component Means by Regime')
plt.ylabel('Mean Value')
plt.tight_layout()
plt.savefig(f'sandbox_output/pc{n_comps}_r{num_reg}_pc_means_barplot.png')

# # This next section will be about factor performance based on the regimes detected


# logging.info('Implementing ANOVA to compare Sharpe ratios across regimes...')
# # Identify regime segments: consecutive rows with the same regime
# daily_factors_with_regime['Regime_Segment'] = (
#     (daily_factors_with_regime['Regime'] != daily_factors_with_regime['Regime'].shift()).cumsum()
# )

# # Store per-regime-instance Sharpe ratios
# segment_sharpes = {factor: [] for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']}
# segment_regimes = []

# # Group by regime segments (i.e., uninterrupted sequences of the same regime)
# for seg_id, segment_data in daily_factors_with_regime.groupby('Regime_Segment'):
#     regime_label = segment_data['Regime'].iloc[0]
    
#     if len(segment_data) < 21:
#         continue  # skip segments too short to compute Sharpe meaningfully
    
#     for factor in segment_sharpes:
#         mean = segment_data[factor].mean()
#         std = segment_data[factor].std()
#         if std != 0 and not np.isnan(std):
#             segment_sharpes[factor].append(mean / std)
#         else:
#             segment_sharpes[factor].append(np.nan)
    
#     segment_regimes.append(regime_label)

# # Run ANOVA on Sharpe ratios per factor
# anova_segment_results = {}
# for factor, values in segment_sharpes.items():
#     df = pd.DataFrame({'Sharpe': values, 'Regime': segment_regimes}).dropna()
#     groups = [df[df['Regime'] == reg]['Sharpe'].values for reg in sorted(df['Regime'].unique())]
    
#     if all(len(g) >= 2 for g in groups):
#         f_stat, p_val = f_oneway(*groups)
#         anova_segment_results[factor] = {'F-statistic': f_stat, 'p-value': p_val}
#     else:
#         anova_segment_results[factor] = {'F-statistic': None, 'p-value': None}



# # Save the results
# anova_segment_df = pd.DataFrame(anova_segment_results).T
# anova_segment_df.to_csv(f'analysis_output/pc{n_comps}_r{num_reg}_anova_segment_sharpe.csv')
# logging.info(f'ANOVA on regime-aligned Sharpe ratios saved as pc{n_comps}_r{num_reg}_anova_segment_sharpe.csv')

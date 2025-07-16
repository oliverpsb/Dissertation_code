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
m_model = MacroPCA(data=macro_df, n_components=4)
m_model.standardise()
pc_df_m = m_model.run_pca()

# Plot the principal components over time
pc_df_m.index = pd.to_datetime(pc_df_m.index)

plt.figure(figsize=(14, 8))
for i in range(4):
    plt.plot(pc_df_m.index, pc_df_m.iloc[:,i], label=f'PC{i+1}')
plt.title('Principal Components Over Time')
plt.xlabel('Date')
plt.ylabel('Component Value')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))  # Every 2 years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  # Rotate x-axis labels for readability

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_timeline.png', dpi=300)  # Save the plot

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

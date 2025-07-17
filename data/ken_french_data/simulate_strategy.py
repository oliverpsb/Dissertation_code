import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def simulate_strategy(merged_df, regime_factors_dict):
    strat_returns = []
    dates = []

    for idx, row in merged_df.iterrows():
        regime = int(row['LaggedRegime'])
        selected_factors = regime_factors_dict.get(regime, [])
        
        # Average return across selected factors
        if selected_factors:
            avg_return = row[selected_factors].mean()
            strat_returns.append(avg_return)
            dates.append(idx)
    
    return pd.Series(strat_returns, index=dates, name='StrategyReturn')

def simulate_sharpe_weighted_strategy(merged_df, sharpe_weights_df):
    returns = []
    dates = []

    for idx, row in merged_df.iterrows():
        regime = int(row['LaggedRegime'])
        if regime not in sharpe_weights_df.index:
            continue  # skip if regime wasn't in Sharpe summary

        weights = sharpe_weights_df.loc[regime].dropna()
        factors = weights.index.tolist()
        weights = weights.values

        factor_values = row[factors].values
        weighted_return = np.dot(factor_values, weights)
        returns.append(weighted_return)
        dates.append(idx)

    return pd.Series(returns, index=dates, name='SharpeWeightedReturn')


def simulate_vol_scaled_strategy(merged_df, regime_factors_dict, rolling_vol_df):
    strat_returns = []
    dates = []

    for idx, row in merged_df.iterrows():
        regime = int(row['LaggedRegime'])
        selected_factors = regime_factors_dict.get(regime, [])

        if selected_factors:
            try:
                vol_row = rolling_vol_df.loc[idx, selected_factors]
                if vol_row.isnull().any():
                    continue

                weights = 1 / vol_row
                weights /= weights.sum()

                factor_returns = row[selected_factors]
                weighted_return = (factor_returns * weights).sum()
                strat_returns.append(weighted_return)
                dates.append(idx)

            except KeyError:
                continue  # Skip early rows with missing vol

    return pd.Series(strat_returns, index=dates, name='VolScaledReturn')


def simulate_filtered_strategy(merged_df, regime_factors_dict, pc_filter_col=None, pc_filter_threshold=0, vix_filter=False, vix_threshold=20):
    strat_returns = []
    dates = []

    for idx, row in merged_df.iterrows():
        regime = int(row['LaggedRegime'])
        selected_factors = regime_factors_dict.get(regime, []).copy()

        # Apply PC1 filter: exclude Mom if PC1 <= threshold
        if pc_filter_col and row[pc_filter_col] <= pc_filter_threshold:
            if 'Mom' in selected_factors:
                selected_factors.remove('Mom')

        # Apply VIX filter: go to cash if VIX > threshold
        if vix_filter and row.get('VIX', 0) > vix_threshold:
            # Go to cash: return RF only
            strat_returns.append(row['RF'])
            dates.append(idx)
            continue

        # Use average return of selected factors + RF (to simulate total return)
        if selected_factors:
            avg_excess = row[selected_factors].mean()
            total_return = avg_excess + row['RF']
            strat_returns.append(total_return)
            dates.append(idx)

    return pd.Series(strat_returns, index=dates, name='FilteredStrategyReturn')


def cumulative_comparison_plot(strategy_returns, benchmark_returns, title):
    strategy_cum = (1 + strategy_returns / 100).cumprod()
    benchmark_cum = (1 + benchmark_returns / 100).cumprod()

    plt.figure(figsize=(14, 8))
    plt.plot(strategy_cum, label='Regime-Based Strategy')
    plt.plot(benchmark_cum, label='Market Benchmark (Mkt-RF)', linestyle='--')
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))  # Every 2 years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)


import matplotlib.pyplot as plt


def compare_all_strategies(strategy_dict, benchmark_series, n_pca_components, title="Cumulative Returns: Strategies vs Benchmark"):
    """
    strategy_dict: dict of {label: pd.Series} for each strategy's return stream (in %)
    benchmark_series: pd.Series of benchmark returns (in %)
    """
    plt.figure(figsize=(14, 7))

    # Compute and plot cumulative returns for each strategy
    for label, returns in strategy_dict.items():
        cumulative = (1 + returns / 100).cumprod()
        plt.plot(cumulative, label=label)

    # Plot benchmark
    benchmark_cum = (1 + benchmark_series / 100).cumprod()
    plt.plot(benchmark_cum, label="Benchmark (Mkt-RF)", linestyle='--', color='black')

    # Plot styling
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))  # Every 2 years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for readability
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PCA_{n_pca_components}_all_strategies.png", dpi=300)

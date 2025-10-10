# Single-Pair Statistical Arbitrage Deep Dive
#
# This script focuses on a detailed analysis of a single cointegrated pair.

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CORE STRATEGY FUNCTIONS ---

def find_best_cointegrated_pair(data):
    """Finds the best cointegrated pair of stocks from a dataframe of price data."""
    n = data.shape[1]
    keys = data.keys()
    best_pair = None
    lowest_pvalue = 0.05

    print("Searching for the best cointegrated pair...")
    for i in range(n):
        for j in range(i + 1, n):
            S1_series = data[keys[i]]
            S2_series = data[keys[j]]
            
            result = coint(S1_series, S2_series)
            pvalue = result[1]
            
            if pvalue < lowest_pvalue:
                lowest_pvalue = pvalue
                best_pair = (keys[i], keys[j], pvalue)

    if best_pair is None: return None
    
    s1_with_const = sm.add_constant(data[best_pair[0]])
    results = sm.OLS(data[best_pair[1]], s1_with_const).fit()
    hedge_ratio = results.params[best_pair[0]]
    
    return {'Stock 1': best_pair[0], 'Stock 2': best_pair[1], 'P-Value': best_pair[2], 'Hedge Ratio': hedge_ratio}


def backtest_strategy_with_adf(data, pair_info, window=252, std_dev_multiplier=2.0, transaction_cost_pct=0.001):
    """
    Performs a walk-forward backtest for a single pair with an ADF check.
    """
    stock1_ticker = pair_info['Stock 1']
    stock2_ticker = pair_info['Stock 2']
    
    print(f"\nBacktesting strategy for {stock1_ticker} and {stock2_ticker} with ADF check...")

    portfolio = pd.DataFrame(index=data.index)
    portfolio['position'] = 0
    
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        
        s1_window = sm.add_constant(window_data[stock1_ticker])
        s2_window = window_data[stock2_ticker]
        model = sm.OLS(s2_window, s1_window).fit()
        hedge_ratio = model.params[stock1_ticker]
        
        spread_window = s2_window - hedge_ratio * s1_window[stock1_ticker]
        adf_pvalue = adfuller(spread_window)[1]

        spread = data[stock2_ticker] - hedge_ratio * data[stock1_ticker]
        rolling_mean = spread.iloc[i-window:i].mean()
        rolling_std = spread.iloc[i-window:i].std()
        current_z_score = (spread.iloc[i] - rolling_mean) / rolling_std
        portfolio.loc[data.index[i], 'z_score'] = current_z_score

        current_position = portfolio.loc[data.index[i-1], 'position']
        
        if current_position == 0 and adf_pvalue <= 0.10:
            if current_z_score > std_dev_multiplier:
                portfolio.loc[data.index[i], 'position'] = -1
            elif current_z_score < -std_dev_multiplier:
                portfolio.loc[data.index[i], 'position'] = 1
        elif (current_position == 1 and current_z_score >= 0) or \
             (current_position == -1 and current_z_score <= 0):
            portfolio.loc[data.index[i], 'position'] = 0
        else:
             portfolio.loc[data.index[i], 'position'] = current_position
             
    portfolio['position'] = portfolio['position'].shift(1).fillna(0)
    trades = portfolio['position'].diff().abs()
    portfolio['transaction_costs'] = trades * transaction_cost_pct
    
    portfolio['stock1_returns'] = data[stock1_ticker].pct_change().fillna(0)
    portfolio['stock2_returns'] = data[stock2_ticker].pct_change().fillna(0)
    
    portfolio['strategy_returns'] = (portfolio['position'] * (portfolio['stock2_returns'] - portfolio['stock1_returns'])) - portfolio['transaction_costs']
    portfolio['cumulative_returns'] = (1 + portfolio['strategy_returns']).cumprod()
    
    return portfolio.dropna()


def calculate_metrics(portfolio):
    """Calculates key performance metrics for a single-pair backtest."""
    daily_returns = portfolio['strategy_returns']
    if daily_returns.std() == 0: return {"Sharpe Ratio": 0, "Maximum Drawdown": 0, "Total Return": 1}
    
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    cumulative_returns = portfolio['cumulative_returns']
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    
    return {
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": drawdown.min(),
        "Total Return": cumulative_returns.iloc[-1]
    }

# --- ADVANCED ANALYSIS FUNCTIONS ---

def run_sensitivity_analysis(data, pair_info, filepath=None):
    """Analyzes the strategy's performance across a range of parameters."""
    print("\n--- Running Sensitivity Analysis ---")
    windows = [60, 120, 252]
    std_devs = [1.5, 2.0, 2.5]
    results = pd.DataFrame(index=windows, columns=std_devs)
    
    for window in windows:
        for std in std_devs:
            portfolio = backtest_strategy_with_adf(data, pair_info, window=window, std_dev_multiplier=std)
            metrics = calculate_metrics(portfolio)
            results.loc[window, std] = metrics['Sharpe Ratio']
    
    results = results.astype(float)
    plt.figure(figsize=(10, 6))
    sns.heatmap(results, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Sharpe Ratio Sensitivity for {pair_info['Stock 1']}-{pair_info['Stock 2']}")
    plt.xlabel('Standard Deviation Multiplier')
    plt.ylabel('Lookback Window (Days)')
    
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Sensitivity heatmap saved to: {filepath}")
    
    plt.show()

def run_market_context_analysis(portfolio, vix_data):
    """Analyzes the strategy's performance by year and VIX regime."""
    print("\n--- Running Market Context Analysis ---")

    def annual_return_func(x):
        return (1 + x).prod() - 1

    def sharpe_ratio_func(x):
        return np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0

    yearly_perf = portfolio['strategy_returns'].groupby(portfolio.index.year).agg(
        **{
            'Annual Return': annual_return_func,
            'Sharpe Ratio': sharpe_ratio_func
        }
    )

    print("\nPerformance by Year:")
    print(yearly_perf.round(2))

    portfolio_with_vix = portfolio.join(vix_data).ffill().dropna()
    portfolio_with_vix['vix_regime'] = pd.qcut(portfolio_with_vix['vix'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    vix_perf = portfolio_with_vix.groupby('vix_regime', observed=True)['strategy_returns'].agg(
        **{'Avg Daily Return': 'mean', 'Sharpe Ratio': sharpe_ratio_func}
    )
    print("\nPerformance by VIX Regime:")
    print(vix_perf.round(4))

# --- VISUALIZATION ---

def plot_results(data, portfolio, pair_info, filepath=None):
    """Plots the primary backtesting results."""
    stock1, stock2 = pair_info['Stock 1'], pair_info['Stock 2']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 2]})
    fig.suptitle(f'Pairs Trading Strategy: {stock1} vs {stock2}', fontsize=16)

    ax1.plot(data[stock1], label=stock1); ax1.plot(data[stock2], label=stock2)
    ax1.set_title('Stock Prices'); ax1.legend(); ax1.grid(True)

    ax2.plot(portfolio['z_score'], label='Z-Score')
    ax2.axhline(2.0, color='r', ls='--'); ax2.axhline(-2.0, color='g', ls='--'); ax2.axhline(0.0, color='k', ls='-')
    ax2.set_title('Spread Z-Score'); ax2.legend(); ax2.grid(True)
    
    ax3.plot(portfolio['cumulative_returns'], label='Strategy Equity Curve')
    ax3.set_title('Portfolio Cumulative Returns'); ax3.legend(); ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Backtest results plot saved to: {filepath}")
        
    plt.show()

# --- MAIN EXECUTION ---

def main():
    if not os.path.exists('img'):
        os.makedirs('img')
        print("Created 'img' directory for plots.")

    tickers = ['PEP', 'KO', 'GS', 'MS', 'JPM', 'AXP', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'UNH', 'JNJ', '^VIX']
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    
    print("Downloading data...")
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna(axis='columns')
        stock_data = raw_data.drop(columns='^VIX')
        vix_data = raw_data[['^VIX']].rename(columns={'^VIX': 'vix'})
    except Exception as e:
        print(f"An exception occurred during data download: {e}"); return

    best_pair = find_best_cointegrated_pair(stock_data)
    if best_pair is None:
        print("No cointegrated pairs found."); return
        
    print(f"\nBest cointegrated pair found: {best_pair['Stock 1']} vs {best_pair['Stock 2']} (p-value: {best_pair['P-Value']:.4f})")
    
    portfolio = backtest_strategy_with_adf(stock_data, best_pair)
    metrics = calculate_metrics(portfolio)
    print("\n--- Primary Backtest Performance (with ADF check) ---")
    for name, val in metrics.items(): print(f"{name}: {val:.2f}")
    
    plot_results(stock_data, portfolio, best_pair, filepath='img/single_pair_results.png')
    run_sensitivity_analysis(stock_data, best_pair, filepath='img/sensitivity_heatmap.png')
    run_market_context_analysis(portfolio, vix_data)

if __name__ == "__main__":
    main()


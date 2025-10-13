# Advanced Multi-Pair Portfolio Backtester with Half-Life Ranking

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CORE UTILITY AND MODELING FUNCTIONS ---

def find_cointegrated_pairs(data):
    """
    Finds cointegrated pairs and calculates their hedge ratios.
    """
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    
    total_pairs = n * (n - 1) // 2
    print(f"Searching for cointegrated pairs... (testing {total_pairs} combinations)")
    
    for i in range(n):
        for j in range(i + 1, n):
            S1_series = data[keys[i]]
            S2_series = data[keys[j]]
            
            result = coint(S1_series, S2_series)
            pvalue = result[1]
            
            if pvalue < 0.05:
                s1_with_const = sm.add_constant(S1_series)
                model = sm.OLS(S2_series, s1_with_const).fit()
                hedge_ratio = model.params[keys[i]]
                pairs.append((keys[i], keys[j], pvalue, hedge_ratio))

    if not pairs: return None
    sorted_pairs = sorted(pairs, key=lambda x: x[2])
    columns = ['Stock 1', 'Stock 2', 'P-Value', 'Hedge Ratio']
    pairs_df = pd.DataFrame(sorted_pairs, columns=columns)
    return pairs_df

def calculate_half_life(spread_series):
    """
    Calculates the half-life of mean reversion for a given spread series
    using the Ornstein-Uhlenbeck process model.
    """
    spread_lagged = spread_series.shift(1).dropna()
    delta_spread = spread_series.diff(1).dropna()
    
    df = pd.concat([delta_spread, spread_lagged], axis=1).dropna()
    df.columns = ['delta_spread', 'spread_lagged']
    
    model = sm.OLS(df['delta_spread'], df['spread_lagged']).fit()
    lambda_ = model.params.iloc[0]
    
    if lambda_ >= 0: return np.inf
    return -np.log(2) / lambda_

def calculate_metrics(portfolio_returns):
    """Calculates key performance metrics for a portfolio."""
    if portfolio_returns.std() == 0: return {"Sharpe Ratio": 0, "Maximum Drawdown": 0, "Total Return": 1}
    
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    
    return {
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": drawdown.min(),
        "Total Return": cumulative_returns.iloc[-1]
    }

# --- ADVANCED PORTFOLIO BACKTESTING ENGINE ---
def backtest_portfolio_with_risk_management(
    data, spy_data, pairs_df, window=252, std_dev_multiplier=2.0, 
    transaction_cost_pct=0.001, max_leverage=1.0, 
    time_stop_days=60, adf_exit_threshold=0.15):
    """
    Backtests a multi-pair portfolio with advanced risk management rules.
    """
    print("\n--- Starting Advanced Portfolio Backtest with Risk Management ---")
    
    pair_names = [f"{p['Stock 1']}-{p['Stock 2']}" for _, p in pairs_df.iterrows()]
    positions = pd.DataFrame(0, index=data.index, columns=pair_names)
    weights = pd.DataFrame(0.0, index=data.index, columns=pair_names)
    portfolio_state = {name: {'days_in_trade': 0} for name in pair_names}
    
    for i in range(window, len(data)):
        date = data.index[i]
        window_data = data.iloc[i-window:i]
        
        # State Tracking and Exit Logic
        for pair_name in pair_names:
            current_position = positions.loc[data.index[i-1], pair_name]
            if current_position != 0:
                portfolio_state[pair_name]['days_in_trade'] += 1
                stock1, stock2 = pair_name.split('-')
                
                s1_window = sm.add_constant(window_data[stock1])
                model = sm.OLS(window_data[stock2], s1_window).fit()
                hedge_ratio = model.params[stock1]
                
                spread_window = window_data[stock2] - hedge_ratio * window_data[stock1]
                spread = data[stock2] - hedge_ratio * data[stock1]
                current_z_score = (spread.iloc[i] - spread_window.mean()) / spread_window.std()
                
                adf_pvalue = adfuller(spread_window)[1]
                if adf_pvalue > adf_exit_threshold or \
                   portfolio_state[pair_name]['days_in_trade'] > time_stop_days or \
                   (current_position == 1 and current_z_score >= 0) or \
                   (current_position == -1 and current_z_score <= 0):
                    positions.loc[date, pair_name] = 0
                    portfolio_state[pair_name]['days_in_trade'] = 0
                else:
                    positions.loc[date, pair_name] = current_position
        
        # Entry Logic
        for _, pair in pairs_df.iterrows():
            stock1, stock2, _, _, _ = pair
            pair_name = f"{stock1}-{stock2}"
            if positions.loc[date, pair_name] == 0:
                s1_window_entry = sm.add_constant(window_data[stock1])
                model_entry = sm.OLS(window_data[stock2], s1_window_entry).fit()
                hedge_ratio = model_entry.params[stock1]

                spread_window = window_data[stock2] - hedge_ratio * window_data[stock1]
                if spread_window.std() == 0: continue
                adf_pvalue = adfuller(spread_window)[1]
                spread = data[stock2] - hedge_ratio * data[stock1]
                current_z_score = (spread.iloc[i] - spread_window.mean()) / spread_window.std()
                
                if abs(current_z_score) > std_dev_multiplier and adf_pvalue <= 0.10:
                    positions.loc[date, pair_name] = -1 if current_z_score > 0 else 1
                    portfolio_state[pair_name]['days_in_trade'] = 1

        # Risk Allocation
        active_positions_for_weighting = {}
        for pair_name in [p for p, pos in positions.loc[date].items() if pos != 0]:
            stock1, stock2 = pair_name.split('-')
            
            s1_window_risk = sm.add_constant(window_data[stock1])
            model_risk = sm.OLS(window_data[stock2], s1_window_risk).fit()
            hedge_ratio = model_risk.params[stock1]

            spread_window = window_data[stock2] - hedge_ratio * window_data[stock1]
            spread = data[stock2] - hedge_ratio * data[stock1]
            active_positions_for_weighting[pair_name] = {
                'z_score': (spread.iloc[i] - spread_window.mean()) / spread_window.std(),
                'volatility': spread_window.std()
            }
        
        if active_positions_for_weighting:
            total_inverse_vol = sum(1 / s['volatility'] for s in active_positions_for_weighting.values())
            base_weights = {name: (1 / s['volatility']) / total_inverse_vol for name, s in active_positions_for_weighting.items()}
            scaled_weights = {name: base_weights[name] * min(abs(s['z_score']) / std_dev_multiplier, 1.5) for name, s in active_positions_for_weighting.items()}
            total_scaled_weight = sum(scaled_weights.values())
            final_weights = {name: w / total_scaled_weight for name, w in scaled_weights.items()}
            
            scaling_factor = max_leverage / sum(final_weights.values()) if sum(final_weights.values()) > max_leverage else 1.0
            for name, weight in final_weights.items():
                weights.loc[date, name] = weight * positions.loc[date, name] * scaling_factor

    # Calculate Returns, Costs, and Hedging
    shifted_weights = weights.shift(1).fillna(0)
    daily_returns = data.pct_change().fillna(0)
    spy_returns = spy_data.pct_change().fillna(0)
    unhedged_portfolio_returns = pd.Series(0.0, index=data.index)
    for pair_name in pair_names:
        stock1, stock2 = pair_name.split('-')
        pair_returns = daily_returns[stock2] - daily_returns[stock1]
        unhedged_portfolio_returns += shifted_weights[pair_name] * pair_returns
        
    rolling_beta = unhedged_portfolio_returns.rolling(window=window).cov(spy_returns) / spy_returns.rolling(window=window).var()
    rolling_beta = rolling_beta.fillna(method='ffill')
    hedged_portfolio_returns = unhedged_portfolio_returns - (rolling_beta.shift(1) * spy_returns)
    turnover = (weights - shifted_weights).abs().sum(axis=1)
    transaction_costs = turnover * transaction_cost_pct
    final_portfolio_returns = hedged_portfolio_returns - transaction_costs
    
    return final_portfolio_returns.dropna(), weights, rolling_beta.dropna()

def plot_portfolio_results(portfolio_returns, weights, beta, filepath=None):
    """Plots the results for the advanced portfolio."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('Advanced Portfolio Performance with Risk Management', fontsize=16)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    ax1.plot(cumulative_returns)
    ax1.set_title('Portfolio Cumulative Returns (Net of Costs & Beta-Hedged)')
    ax1.grid(True)
    leverage = weights.abs().sum(axis=1)
    ax2.plot(leverage)
    ax2.set_title('Portfolio Gross Leverage'); ax2.axhline(leverage.mean(), color='r', ls='--', label=f'Avg Leverage: {leverage.mean():.2f}'); ax2.legend(); ax2.grid(True)
    ax3.plot(beta)
    ax3.set_title('Portfolio Rolling Beta to SPY'); ax3.axhline(0, color='k', ls='--'); ax3.axhline(beta.mean(), color='r', ls='--', label=f'Avg Beta: {beta.mean():.2f}'); ax3.legend(); ax3.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Portfolio results plot saved to: {filepath}")
        
    plt.show()

# --- MAIN EXECUTION ---
def main():
    if not os.path.exists('img'):
        os.makedirs('img')
        print("Created 'img' directory for plots.")

    tickers = ['PEP', 'KO', 'GS', 'MS', 'JPM', 'AXP', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'UNH', 'JNJ', 'SPY']
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    
    print("Downloading data...")
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna(axis='columns')
        stock_data = raw_data.drop(columns='SPY')
        spy_data = raw_data['SPY']
    except Exception as e:
        print(f"An exception occurred during data download: {e}"); return

    # --- Step 1: Filter and Rank Pairs ---
    coint_pairs_df = find_cointegrated_pairs(stock_data)
    if coint_pairs_df is None or coint_pairs_df.empty: print("No cointegrated pairs found."); return
        
    print("\nCalculating mean reversion half-life for each pair...")
    half_lives = []
    for idx, row in coint_pairs_df.iterrows():
        spread = stock_data[row['Stock 2']] - row['Hedge Ratio'] * stock_data[row['Stock 1']]
        hl = calculate_half_life(spread)
        half_lives.append(hl)
        
    coint_pairs_df['Half-Life (Days)'] = half_lives
    coint_pairs_df = coint_pairs_df.sort_values(by='Half-Life (Days)')
    
    print("\n--- Cointegrated Pairs Ranked by Half-Life ---")
    print(coint_pairs_df.to_string())
    
    # --- Step 2: Select the best pairs and run the backtest ---
    top_5_pairs = coint_pairs_df.head(5)
    print("\n--- Selected Top 5 Pairs for Portfolio ---")
    print(top_5_pairs)

    portfolio_returns, weights, beta = backtest_portfolio_with_risk_management(stock_data, spy_data, top_5_pairs)
    
    metrics = calculate_metrics(portfolio_returns)
    print("\n--- Advanced Portfolio Performance ---")
    for name, val in metrics.items(): print(f"{name}: {val:.2f}")

    plot_portfolio_results(portfolio_returns, weights, beta, filepath='img/portfolio_results.png')

if __name__ == "__main__":
    main()


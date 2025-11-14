# Multi-Pair Portfolio Backtester with Walk-Forward Selection (DJIA)
#
# The architecture is a full walk forward analysis:
# 1. The simulation is broken into 3-year "chunks" (e.g., 2015-17, 2018-20, 2021-23).
# 2. Before each chunk, the model runs its entire pair-selection process
#    (Sector filter, Cointegration, Half-Life rank) using only the
#    data from the preceding 3 years (the "lookback period").
# 3. The backtest then trades this newly-selected portfolio
#    for the next 3 years.

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.tseries.offsets import DateOffset

# --- ECONOMIC FILTERING FUNCTION ---

def get_sector_map(tickers):
    """
    Fetches the GICS sector for each ticker and returns a dictionary map.
    """
    print("Fetching GICS sector data for all tickers...")
    sector_map = {}
    count = 0
    for ticker in tickers:
        count += 1
        try:
            t = yf.Ticker(ticker)
            sector = t.info.get('sector')
            
            if sector:
                sector_map[ticker] = sector
                print(f"  ({count}/{len(tickers)}) {ticker}: {sector}")
            else:
                print(f"  ({count}/{len(tickers)}) {ticker}: No sector info found.")
        except Exception as e:
            print(f"  ({count}/{len(tickers)}) {ticker}: Error fetching info, skipping.")
            
    return sector_map

# --- CORE UTILITY AND MODELING FUNCTIONS ---

def find_cointegrated_pairs(data, sector_map):
    """
    Finds cointegrated pairs only within the same sector,
    and then calculates their hedge ratios.
    """
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    
    total_pairs = n * (n - 1) // 2
    tested_pairs_count = 0
    print(f"\nSearching for cointegrated pairs... (Total combinations: {total_pairs})")
    print("Applying economic sector filter...")
    
    for i in range(n):
        for j in range(i + 1, n):
            stock1 = keys[i]
            stock2 = keys[j]
            
            if stock1 not in sector_map or stock2 not in sector_map:
                continue
            if sector_map[stock1] != sector_map[stock2]:
                continue
            
            tested_pairs_count += 1
            S1_series = data[stock1]
            S2_series = data[stock2]
            
            # Ensure we have enough non-nan data
            if S1_series.isnull().any() or S2_series.isnull().any():
                continue
            
            result = coint(S1_series, S2_series)
            pvalue = result[1]
            
            if pvalue < 0.05:
                s1_with_const = sm.add_constant(S1_series)
                model = sm.OLS(S2_series, s1_with_const).fit()
                hedge_ratio = model.params[stock1]
                pairs.append((stock1, stock2, pvalue, hedge_ratio))

    print(f"Economically-valid pairs tested: {tested_pairs_count}")
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
    data, spy_data, pairs_df, 
    trade_start_date, trade_end_date,
    window=252, std_dev_multiplier=2.0, 
    transaction_cost_pct=0.001, max_leverage=1.0, 
    time_stop_days=60, adf_exit_threshold=0.15):
    """
    Backtests a multi-pair portfolio for a specific time chunk.
    """
    print(f"\n--- Backtesting from {trade_start_date} to {trade_end_date} ---")
    
    pair_names = [f"{p['Stock 1']}-{p['Stock 2']}" for _, p in pairs_df.iterrows()]
    positions = pd.DataFrame(0, index=data.index, columns=pair_names)
    weights = pd.DataFrame(0.0, index=data.index, columns=pair_names)
    hedge_ratios_df = pd.DataFrame(0.0, index=data.index, columns=pair_names)
    portfolio_state = {name: {'days_in_trade': 0} for name in pair_names}
    
    try:
        start_date_ts = pd.to_datetime(trade_start_date)
        end_date_ts = pd.to_datetime(trade_end_date)
        
        # Get the first valid trading day on or after* the start date
        start_index_date = data.index[data.index >= start_date_ts][0]
        # Get the last valid trading day on or before* the end date
        end_index_date = data.index[data.index <= end_date_ts][-1]
        
        # Get the integer locations of these dates
        start_index = data.index.get_loc(start_index_date)
        end_index = data.index.get_loc(end_index_date)

    except (KeyError, IndexError) as e:
        print(f"Error: Date range {trade_start_date}-{trade_end_date} not found in data index. Skipping chunk. {e}")
        return pd.Series(dtype=float), pd.DataFrame(dtype=float), pd.Series(dtype=float)
        
    if start_index < window:
        print(f"Trade start date {trade_start_date} is too early.")
        start_index = window

    for i in range(start_index, end_index + 1):
        date = data.index[i]
        window_data = data.iloc[i-window:i]
        
        daily_stats = {}
        for pair_name in pair_names:
            stock1, stock2 = pair_name.split('-')
            
            s1_window = sm.add_constant(window_data[stock1])
            s2_window = window_data[stock2]
            model = sm.OLS(s2_window, s1_window).fit()
            hedge_ratio = model.params[stock1]
            
            hedge_ratios_df.loc[date, pair_name] = hedge_ratio
            
            spread_window = s2_window - hedge_ratio * s1_window[stock1]
            spread_volatility = spread_window.std()

            if spread_volatility == 0:
                daily_stats[pair_name] = {'adf_pvalue': 1, 'z_score': 0, 'volatility': 0}
                continue
            
            spread = data[stock2] - hedge_ratio * data[stock1]
            daily_stats[pair_name] = {
                'adf_pvalue': adfuller(spread_window)[1],
                'z_score': (spread.iloc[i] - spread_window.mean()) / spread_volatility,
                'volatility': spread_volatility
            }

        # 1. State Tracking and Exit Logic
        for pair_name in pair_names:
            current_position = positions.loc[data.index[i-1], pair_name]
            if current_position != 0:
                portfolio_state[pair_name]['days_in_trade'] += 1
                stats = daily_stats[pair_name]
                
                if stats['adf_pvalue'] > adf_exit_threshold or \
                   portfolio_state[pair_name]['days_in_trade'] > time_stop_days or \
                   (current_position == 1 and stats['z_score'] >= 0) or \
                   (current_position == -1 and stats['z_score'] <= 0):
                    positions.loc[date, pair_name] = 0
                    portfolio_state[pair_name]['days_in_trade'] = 0
                else:
                    positions.loc[date, pair_name] = current_position
        
        # 2. Entry Logic
        for _, pair in pairs_df.iterrows():
            stock1, stock2, _, _, _ = pair
            pair_name = f"{stock1}-{stock2}"
            if positions.loc[date, pair_name] == 0:
                stats = daily_stats[pair_name]
                if abs(stats['z_score']) > std_dev_multiplier and stats['adf_pvalue'] <= 0.10:
                    positions.loc[date, pair_name] = -1 if stats['z_score'] > 0 else 1
                    portfolio_state[pair_name]['days_in_trade'] = 1

        # 3. Risk Allocation
        active_positions_for_weighting = {}
        for pair_name in [p for p, pos in positions.loc[date].items() if pos != 0]:
            if daily_stats[pair_name]['volatility'] > 0:
                active_positions_for_weighting[pair_name] = daily_stats[pair_name]
        
        if active_positions_for_weighting:
            total_inverse_vol = sum(1 / s['volatility'] for s in active_positions_for_weighting.values())
            base_weights = {name: (1 / s['volatility']) / total_inverse_vol for name, s in active_positions_for_weighting.items()}
            scaled_weights = {name: base_weights[name] * min(abs(s['z_score']) / std_dev_multiplier, 1.5) for name, s in active_positions_for_weighting.items()}
            
            total_scaled_weight = sum(scaled_weights.values())
            if total_scaled_weight == 0: continue
            final_proportions = {name: w / total_scaled_weight for name, w in scaled_weights.items()}

            for name, proportion in final_proportions.items():
                direction = positions.loc[date, name]
                weights.loc[date, name] = proportion * direction * max_leverage

    trade_period_data = data.loc[trade_start_date:trade_end_date]
    
    shifted_weights = weights.loc[trade_period_data.index].shift(1).fillna(0)
    shifted_hedge_ratios = hedge_ratios_df.loc[trade_period_data.index].shift(1).fillna(0)
    
    daily_returns = data.loc[trade_period_data.index].pct_change().fillna(0)
    spy_returns = spy_data.loc[trade_period_data.index].pct_change().fillna(0)
    unhedged_portfolio_returns = pd.Series(0.0, index=trade_period_data.index)
    
    for pair_name in pair_names:
        stock1, stock2 = pair_name.split('-')
        hedge_ratio = shifted_hedge_ratios[pair_name]
        pair_returns = daily_returns[stock2] - (hedge_ratio * daily_returns[stock1])
        unhedged_portfolio_returns += shifted_weights[pair_name] * pair_returns
        
    rolling_beta = unhedged_portfolio_returns.rolling(window=window).cov(spy_returns) / spy_returns.rolling(window=window).var()
    rolling_beta = rolling_beta.fillna(method='ffill')
    hedged_portfolio_returns = unhedged_portfolio_returns - (rolling_beta.shift(1) * spy_returns)
    
    turnover = (weights.loc[trade_period_data.index] - shifted_weights).abs().sum(axis=1)
    transaction_costs = turnover * transaction_cost_pct
    final_portfolio_returns = hedged_portfolio_returns - transaction_costs
    
    return final_portfolio_returns.dropna(), weights.loc[trade_period_data.index], rolling_beta.dropna()

def plot_portfolio_results(portfolio_returns, weights, beta, filepath=None):
    """Plots the results for the advanced portfolio."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('Walk-Forward Selection Portfolio Performance (DJIA Universe)', fontsize=16) # <-- MODIFIED
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

    tickers = [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MRK',
        'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'CVX',
        'SPY' # Add SPY for hedging
    ]
    tickers = sorted(list(set(tickers))) # Remove duplicates
    
    start_date = '2012-01-01'
    end_date = '2023-12-31'
    
    print(f"\nDownloading data for {len(tickers)} DJIA tickers from {start_date}...")
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        raw_data_cleaned = raw_data.dropna(axis='columns')
        
        if 'SPY' not in raw_data_cleaned.columns:
            print("Error: SPY data is missing or incomplete. Cannot proceed."); return
        
        stock_data = raw_data_cleaned.drop(columns='SPY')
        spy_data = raw_data_cleaned['SPY']
        print(f"Data download complete. {len(stock_data.columns)} stocks have full data.")
    except Exception as e:
        print(f"An exception occurred during data download: {e}"); return

    sector_map = get_sector_map(stock_data.keys())
    
    # --- Define Lookback and Trading Chunks ---
    lookback_periods = [('2012-01-01', '2014-12-31'),
                        ('2015-01-01', '2017-12-31'),
                        ('2018-01-01', '2020-12-31')]
    
    trade_periods = [('2015-01-01', '2017-12-31'),
                     ('2018-01-01', '2020-12-31'),
                     ('2021-01-01', '2023-12-31')]

    all_returns = []
    all_weights = []
    all_betas = []

    # --- Loop through each recalibration chunk ---
    for (lb_start, lb_end), (trade_start, trade_end) in zip(lookback_periods, trade_periods):
        print(f"\n--- Running Selection for {trade_start} to {trade_end} ---")
        print(f"Using lookback data from {lb_start} to {lb_end}")
        
        # 1. Filter and rank pairs using only the lookback data
        lookback_data = stock_data.loc[lb_start:lb_end]
        coint_pairs_df = find_cointegrated_pairs(lookback_data, sector_map)
        
        if coint_pairs_df is None or coint_pairs_df.empty: 
            print("No pairs found for this period, skipping chunk."); continue
            
        half_lives = [calculate_half_life(lookback_data[row['Stock 2']] - row['Hedge Ratio'] * lookback_data[row['Stock 1']]) for _, row in coint_pairs_df.iterrows()]
        coint_pairs_df['Half-Life (Days)'] = half_lives
        coint_pairs_df = coint_pairs_df.sort_values(by='Half-Life (Days)')
        
        top_5_pairs = coint_pairs_df.head(5)
        print(f"Top 5 pairs for {trade_start}-{trade_end}:")
        print(top_5_pairs)
        
        if top_5_pairs.empty:
            print(f"No valid pairs found for chunk {trade_start}-{trade_end}, skipping.")
            continue

        # 2. Run the backtest for the trade period using these pairs
        # We pass the full data, but specify the trade chunk
        returns, weights, beta = backtest_portfolio_with_risk_management(
            stock_data, spy_data, top_5_pairs, 
            trade_start_date=trade_start, 
            trade_end_date=trade_end,
            window=252 # Use a 1-year rolling window
        )
        
        all_returns.append(returns)
        all_weights.append(weights)
        all_betas.append(beta)

    # --- Stitch all the performance chunks together ---
    if not all_returns:
        print("No results generated.")
        return
        
    final_portfolio_returns = pd.concat(all_returns)
    final_weights = pd.concat(all_weights)
    final_beta = pd.concat(all_betas)

    # --- Final Analysis ---
    metrics = calculate_metrics(final_portfolio_returns)
    print("\n--- Final Walk-Forward Portfolio Performance (DJIA, 2015-2023) ---")
    for name, val in metrics.items(): print(f"{name}: {val:.2f}")

    plot_portfolio_results(final_portfolio_returns, final_weights, final_beta, filepath='img/portfolio_results_walk_forward_djia.png')

if __name__ == "__main__":
    main()
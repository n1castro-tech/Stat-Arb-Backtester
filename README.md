# **Statistical Arbitrage & Pairs Trading Research Engine**

This repository contains the code for a statistical arbitrage (pairs trading) strategy, demonstrating a research workflow that evolves from a single-pair analysis to a sophisticated, multi-asset portfolio.  
This project is organized into two distinct scripts, each with a clear purpose:

1. **single\_pair\_deep\_dive.py**: Focuses on a deep-dive analysis of a single cointegrated pair.  
2. **multi\_pair\_portfolio\_backtest.py**: Implements a full-scale portfolio simulation with advanced risk management and a superior method for pair selection.

## **Part 1: Single-Pair Deep Dive (single\_pair\_deep\_dive.py)**

This initial phase focuses on building a robust backtester for a single cointegrated pair. The goal is to go beyond a simple performance metric and understand the strategy's sensitivities and failure points.

* **Core Logic:** Implements a walk-forward backtest using an Engle-Granger test to identify a candidate pair.  
* **Regime Stability:** Incorporates a dynamic Augmented Dickey-Fuller (ADF) test to ensure trades are only placed when the pair's relationship is statistically stable.  
* **Advanced Analysis:**  
  * **Sensitivity Analysis:** Generates heatmaps to show how the strategy's Sharpe Ratio responds to changes in key parameters (lookback window, z-score threshold).  
  * **Market Context Analysis:** Breaks down the strategy's performance by year and by market volatility (VIX) regimes.

## **Part 2: Multi-Pair Portfolio Backtest (multi\_pair\_portfolio\_backtest.py)**

This script scales the strategy to a multi-asset portfolio and introduces a superior method for pair selection, along with a full suite of professional-grade risk management tools.

* **Filter & Rank Methodology:**  
  1. **Filter:** First, the engine uses the Engle-Granger test to find a broad universe of all cointegrated pairs.  
  2. **Rank:** It then models each pair's spread using an **Ornstein-Uhlenbeck process** to calculate the **half-life of mean reversion**.  
* **Portfolio Construction:** Trades a basket of the top 5 pairs with the shortest half-lives. This ensures capital is deployed to the opportunities with the most attractive, tradable dynamics.  
* **Advanced Risk Management:**  
  * **Dynamic Hedge Ratios:** The model adapts to changing market conditions by using a rolling hedge ratio for all calculations.  
  * **Statistical & Time-Based Stops:** Adds thesis-driven stop-losses to exit trades if the statistical relationship weakens or a trade takes too long to revert.  
  * **Gross Exposure Limits:** Enforces a hard cap on the portfolio's total leverage.  
  * **Beta Hedging:** Neutralizes the portfolio's exposure to overall market movements using an S\&P 500 ETF (SPY), isolating the strategy's alpha.

## **How to Run**

1. Ensure you have Python and the required libraries installed: pandas, yfinance, numpy, statsmodels, seaborn, matplotlib.  
2. Clone this repository.  
3. Run either script from your terminal:  
   \# To run the single-pair deep dive:  
   python single\_pair\_deep\_dive.py

   \# To run the full portfolio backtest:  
   python multi\_pair\_portfolio\_backtest.py  

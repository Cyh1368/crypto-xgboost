# Crypto-XGBoost: 15-Minute Futures Trading Algorithm

## Project Purpose
The goal of this project is to develop and backtest a high-performance trading algorithm for cryptocurrency futures using the **XGBoost** gradient boosting framework. The strategy operates on a **15-minute timeframe**, leveraging multi-asset data and advanced feature engineering to predict short-term price movements and execute trades with high precision.

## Model Structure & Features
The core of the strategy is an XGBoost regressor trained to predict the next-bar basis point (BPS) return.

### Model Configuration:
- **Framework**: XGBoost (Extreme Gradient Boosting)
- **Timeframe**: 15 Minutes
- **Objective**: Regression (minimize RMSE/MAE of predicted vs actual returns)
- **Target**: Next bar BPS change: `(Close_{t+1} / Close_t - 1) * 10,000`

### Feature Set (47 Total Features):
The model utilizes a comprehensive set of features categorized into:
1.  **Orderbook Microstructure**:
    *   **OBI (Order Book Imbalance)**: Calculated at multiple decay factors (tau=1, 3, 5, 10).
    *   **Depth Ratios**: Bid/Ask volume ratios at different levels.
    *   **Book Pressure**: Mid-level volume pressure.
    *   **Spread Analysis**: Real-time spread in basis points.
2.  **Price Action & Momentum**:
    *   **Returns**: Log returns at various lags (1, 3, 6, 12, 48 bars).
    *   **Volatility**: Rolling standard deviations (5, 20, 60 bars).
    *   **Technical Indicators**: RSI (6, 14), MACD, Bollinger Bands, ATR.
    *   **Bar Characteristics**: Momentum, Wick Ratios (Up/Down).
3.  **Advanced Market Dynamics**:
    *   **Kyle's Lambda Estimation**: Estimating market impact and liquidity.
    *   **VWAP Deviation**: Distance from Volume Weighted Average Price.
    *   **Statistical Moments**: Skewness and Kurtosis (20 bars).
4.  **Macro & Session Features**:
    *   **Funding Rates**: Current and 8-hour moving average.
    *   **Temporal Features**: Sine/Cosine encodings for Hour of Day and Day of Week.
    *   **Session Indicators**: Binary markers for Asia/US sessions and weekends.

## Current Backtest Results
The model was tested across 10 major crypto pairs. The results demonstrate significant alpha generation and high risk-adjusted returns (Sharpe/Sortino).

| Symbol | Total Net PnL ($) | Win Rate (%) | Profit Factor | Sharpe Ratio | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DOT_USDT** | **38,668.40** | **73.43%** | **6.545** | **57.839** | **-0.18%** |
| ADA_USDT | 29,219.20 | 70.86% | 6.430 | 51.917 | -0.23% |
| DOGE_USDT | 25,025.80 | 70.59% | 6.074 | 46.325 | -0.14% |
| LINK_USDT | 22,680.90 | 70.78% | 5.362 | 42.181 | -0.22% |
| ETH_USDT | 17,567.50 | 68.00% | 5.101 | 34.199 | -0.13% |
| XRP_USDT | 8,765.70 | 68.58% | 3.722 | 24.300 | -0.18% |
| SOL_USDT | 7,843.31 | 54.91% | 1.884 | 13.940 | -0.48% |
| BTC_USDT | 5,058.07 | 64.80% | 3.070 | 16.163 | -0.15% |
| LTC_USDT | 3,407.45 | 58.21% | 3.204 | 14.489 | -0.15% |
| BNB_USDT | -29.47 | 47.93% | 0.984 | -3.793 | -0.94% |

### DOT-USDT Advanced Backtest Visualization
Below is the advanced performance visualization for Polkadot (DOT), showcasing the PnL curve, trade execution timing, and predictive accuracy distribution.

![DOT_USDT Advanced Backtest](backtester_v1/results/advanced_backtest_DOT_USDT.png)

## Repository Structure
- `backtester_v1/scripts/`: Core execution scripts (Backtesting, Training, Report Generation).
- `backtester_v1/models/`: Trained XGBoost models and scalers.
- `backtester_v1/results/`: Detailed reports, interactive HTML visualizations, and performance plots.
- `References/`: Project documentation and guides.

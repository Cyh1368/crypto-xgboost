# `Crypto-XGBoost`: 15-Minute Perps Trading Algorithm

`Crypto-XGBoost` is a trading algorithm built upon the [XGBoost](https://arxiv.org/abs/1603.02754) framework. Trained on 10 cryptocurrencies, it predicts the price in 15 minutes, and places orders based on a trading logic developed through [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve). We are ready to implement this strategy in real time.

## Current Backtest Results
The model was tested across 10 major crypto pairs. The results demonstrate significant alpha generation and high risk-adjusted returns (Sharpe/Sortino).

| Symbol | Total Net PnL ($) | Win Rate (%) | Profit Factor | Sharpe Ratio | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DOT_USDT** | **82,910.80** | **85.73%** | **22.418** | **60.648** | **-0.13%** |
| ADA_USDT | 64,219.20 | 85.67% | 19.486 | 55.151 | -0.15% |
| DOGE_USDT | 57,511.00 | 84.88% | 20.442 | 50.951 | -0.10% |
| LINK_USDT | 51,535.10 | 84.79% | 18.433 | 47.161 | -0.18% |
| BTC_USDT | 21,534.70 | 79.86% | 16.265 | 32.833 | -0.07% |
| ETH_USDT | 22,768.60 | 83.94% | 16.930 | 28.629 | -0.14% |
| LTC_USDT | 15,197.70 | 76.18% | 11.706 | 26.935 | -0.13% |
| XRP_USDT | 11,305.80 | 85.15% | 24.231 | 25.344 | -0.07% |
| SOL_USDT | 21,304.20 | 66.26% | 3.153 | 22.859 | -1.27% |
| BNB_USDT | 2,332.14 | 65.82% | 4.296 | 7.079 | -0.16% |

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

### DOT-USDT Advanced Backtest Visualization
Below is the advanced performance visualization for Polkadot (DOT), showcasing the PnL curve, trade execution timing, strategy position, and predictive accuracy distribution.

![DOT_USDT Advanced Backtest](backtester_v1/results/advanced_backtest_DOT_USDT.png)

## Repository Structure
- `backtester_v1/scripts/`: Core execution scripts (Backtesting, Training, Report Generation).
- `backtester_v1/models/`: Trained XGBoost models and scalers.
- `backtester_v1/results/`: Detailed reports, interactive HTML visualizations, and performance plots.
- `References/`: Project documentation and guides.

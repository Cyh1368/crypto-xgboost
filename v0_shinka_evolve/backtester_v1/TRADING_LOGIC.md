# Trading Strategy Logic — ShinkáEvolve v0

This document details the signal generation, risk management, and execution logic implemented in the `backtester_v1` system.

## 1. Signal Pipeline

The strategy operates on 15-minute bars using a pre-trained XGBoost regression model.

### 1.1 Model Inference & Calibration
1.  **Feature Input**: Every 15 minutes, a 47-feature vector is generated from OHLCV and L2 orderbook data.
2.  **Raw Prediction**: The XGBoost model outputs a raw prediction in Basis Points (BPS), representing the expected price change for the next bar.
3.  **Variance Calibration**: To correct for the inherent "regression to the mean" in statistical models, a calibration factor of **1.0411** is applied to the raw BPS output.
4.  **Ratio Conversion**: The calibrated BPS is converted into a price ratio:
    $$Predicted\_Ratio = 1.0 + \left(\frac{BPS \times 1.0411}{10,000}\right)$$

### 1.2 Directional Thresholding
The strategy uses a "dead zone" to filter out low-conviction signals and noise.
-   **LONG**: Triggered if $Predicted\_Ratio > 1.000080$ (approx. +0.8 BPS).
-   **SHORT**: Triggered if $Predicted\_Ratio < 0.999920$ (approx. -0.8 BPS).
-   **FLAT**: If the prediction falls within the range $[0.999920, 1.000080]$, no trade is entered, and any existing position is closed.

---

## 2. Risk Management Filters

Before a signal is executed, it must pass three primary risk filters designed to protect capital in adverse market conditions.

| Filter | Logic | Rationale |
| :--- | :--- | :--- |
| **Liquidity Filter** | `spread_bps > 10` | Prevents entry when the bid-ask spread is too wide, as high slippage would erode the expected BPS alpha. |
| **Volatility Filter** | `vol_5 > 3 * vol_60` | Circuit breaker that halts trading if short-term volatility (5 bars) spikes significantly above the hourly trend (60 bars). |
| **Funding Bias** | `abs(funding_rate) > 0.001` | Suppresses LONGs when funding is deeply negative (bearish bias) and SHORTs when funding is highly positive (bullish bias). |

---

## 3. Position and Execution

### 3.1 Sizing and Leverage
-   **Fixed Notional**: The strategy targets a fixed position size of **$10,000 USD** per trade.
-   **Leverage Limit**: Total position notional is capped at **3.0x** of the current account equity.
-   **Account Equity**: Backtest starts with an initial equity of **$100,000 USD**.

### 3.2 Fees and Slippage
-   **Fee Model**: A flat **5 BPS (0.0005)** taker fee is applied to both entry and exit orders.
-   **Execution**: Trades are executed at the `close` price of the bar when the signal is generated.

---

## 4. Performance Summary (Validation Results)

The following results were achieved on the **5,000-bar Kraken BTC/USDT** out-of-sample validation set.

| Metric | Value |
| :--- | :--- |
| **Total Net PnL ($)** | **705.99** |
| **Total Trades** | 40 |
| **Win Rate (%)** | **72.5%** |
| **Profit Factor** | 5.752 |
| **Sharpe Ratio (annual)** | **19.707** |
| **Sortino Ratio** | 54.682 |
| **Max Drawdown (%)** | **-0.07%** |
| **Calmar Ratio** | 9.8 |
| **Avg Win ($)** | 29.47 |
| **Avg Loss ($)** | -13.51 |
| **Total Fees Paid ($)** | 400.04 |

---
*Note: Results are based on event-driven simulation on historical data. Live performance may be subject to additional latency and order book impact.*

# BTC Futures XGBoost Alpha Signal

This repository contains a production-grade XGBoost-driven regression model for predicting 15-minute Bitcoin price ratios (Price_{t+1} / Price_t).

## 🚀 Overview

The model uses a "kitchen sink" approach to feature engineering, combining orderbook microstructure, price action, and cyclical time features to capture alpha in the 15-minute timeframe.

## 📊 Features

### 1. Orderbook Microstructure
- **Exponentially-Decayed OBI**: Weighted volume imbalance across the top 20 levels using decay factors $\tau \in \{1, 3, 5, 10\}$.
- **Spread & Depth**: Bid-ask spread in BPS, depth ratios at 5/10 levels, and book pressure.
- **Micro-movements**: Mid-price moves and Kyle's Lambda estimation ($\Delta P / \Delta V$).

### 2. Price Action & Momentum
- **Log Returns**: Multi-period returns (1, 3, 6, 12, 48 bars).
- **Volatility**: Rolling standard deviations (5, 20, 60 bars) and realized volatility ratios.
- **Technical Indicators**: RSI (6, 14), MACD Signal, Bollinger Band % position, and normalized ATR.
- **Bar Microstructure**: Momentum ratio, wick ratios (up/down), and volume-to-MA ratios.

### 3. Time & Session
- **Cyclical Encoding**: Sine/Cosine transforms for Hour-of-Day and Day-of-Week.
- **Session Binaries**: Asia/US session flags and weekend detection.
- **Funding Proximity**: Minutes remaining until the next 8-hour funding payment.

## 🏗️ Model Training

- **Architecture**: XGBoost Regressor with `reg:squarederror` objective.
- **Data Source**: Aggregated real 15-minute OHLCV + L2 Proxy data from 10 major crypto pairs (BTC, ETH, SOL, ADA, etc.).
- **Scaling**: Robust `StandardScaler` applied to all non-binary features.
- **Optimization**: Early stopping on RMSE to prevent overfitting.

## 📈 Performance Results (Calibrated)

The model was updated to predict BPS changes and calibrated with a variance-scaling factor (**1.0411**) to fix under-prediction bias.

| Metric | Value |
|---|---|
| **Test Set Correlation (Multi-coin)** | **0.7286** |
| **Validation Correlation (BTC Fresh)** | **0.7197** |
| **Calibration Factor** | 1.0411 |
| **Predicted Ratio Mean** | 1.000165 |
| **Predicted Ratio Std** | 0.001053 |

### Visualizations
Plots can be found in the `results/` directory:
- `final_ratio_scatter.png`: Training/Test scatter plot.
- `validation_scatter.png`: Out-of-sample 5000-bar validation plot.

## 🛠️ Usage

### Training
```bash
python btc_backtester/scripts/train_final_model.py
```

### Validation
```bash
python btc_backtester/scripts/validate_model.py
```

---
*Note: This is part of the ShinkáEvolve project loop.*

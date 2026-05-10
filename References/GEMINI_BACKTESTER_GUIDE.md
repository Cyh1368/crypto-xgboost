# XGBoost BTC Alpha — Backtester Implementation Guide
### For Gemini CLI (ShinkáEvolve v0)

> **Model**: `xgb_regression_v0.json` — XGBoost GBTree, 968 trees, 47 features  
> **Target**: 15-minute BTC price ratio (`Price_{t+1} / Price_t`) predicted as BPS change  
> **Calibration factor**: `1.0411` (post-hoc variance scaling)  
> **Validated OOS correlation**: `0.7197` on 5,000-bar Kraken BTC data

---

## 0. Your Task (Gemini CLI Instructions)

You are implementing a **self-contained Python backtester** that:

1. Loads the pre-trained XGBoost model (`xgb_regression_v0.json`)
2. Ingests historical 15-minute OHLCV + L2 orderbook proxy data
3. Reconstructs all **47 features** identically to training
4. Runs inference to get calibrated ratio predictions bar-by-bar
5. Applies a **threshold-based directional trading strategy** on those predictions
6. Computes full performance metrics and outputs a report

Do **not** retrain the model. Do **not** change the feature names or order. The model expects exactly the 47 features listed in Section 3, in the exact order given.

---

## 1. Project Structure

Create the following directory layout:

```
btc_backtester/
├── data/
│   ├── raw/                    # Place your OHLCV + orderbook CSVs here
│   └── processed/              # Feature matrices written here
├── models/
│   └── xgb_regression_v0.json  # The pre-trained model (copy here)
├── results/
│   ├── trades.csv
│   ├── equity_curve.csv
│   └── backtest_report.md
├── scripts/
│   ├── feature_engineering.py
│   ├── backtester.py
│   └── report.py
├── requirements.txt
└── run_backtest.py             # Main entry point
```

---

## 2. Dependencies

**`requirements.txt`**:
```
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.3.0
ta>=0.11.0
matplotlib>=3.8.0
scipy>=1.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 3. Feature Engineering (`scripts/feature_engineering.py`)

> **CRITICAL**: The model was trained with a `StandardScaler` on all non-binary features. You must apply the same scaling. If you do not have the saved scaler, fit a `RobustScaler` on the **training portion** of your data only, then transform the full dataset. Never fit on the full dataset.

The model uses exactly **47 features** in this order:

```python
FEATURE_NAMES = [
    # --- A. Orderbook Microstructure ---
    'obi_tau1',       # Exp-decayed OBI, tau=1, top 20 levels
    'obi_tau3',       # Exp-decayed OBI, tau=3
    'obi_tau5',       # Exp-decayed OBI, tau=5
    'obi_tau10',      # Exp-decayed OBI, tau=10
    'spread_bps',     # (Ask1 - Bid1) / Mid * 10000
    'depth_ratio_5',  # BidVol[1:5] / AskVol[1:5]
    'depth_ratio_10', # BidVol[1:10] / AskVol[1:10]
    'mid_price_move', # (Mid_t - Mid_{t-1}) / Mid_{t-1}
    'book_pressure_3',# BidVol[2:4] / AskVol[2:4]
    'kyle_lambda_est',# abs(delta_price) / abs(delta_volume)

    # --- B. Price Action & Momentum ---
    'ret_1',          # log(close_t / close_{t-1})
    'ret_3',          # log(close_t / close_{t-3})
    'ret_6',          # log(close_t / close_{t-6})
    'ret_12',         # log(close_t / close_{t-12})
    'ret_48',         # log(close_t / close_{t-48})
    'vol_5',          # Rolling std of ret_1, window=5
    'vol_20',         # Rolling std of ret_1, window=20
    'vol_60',         # Rolling std of ret_1, window=60
    'rsi_14',         # RSI(14) / 100
    'rsi_6',          # RSI(6) / 100
    'macd_signal',    # MACD signal line (12,26,9 EMA)
    'bb_pct',         # (close - BB_lower) / (BB_upper - BB_lower), window=20
    'atr_14',         # ATR(14) normalized by close
    'momentum_bar',   # (close - open) / (high - low + 1e-9)
    'wick_ratio_up',  # (high - max(open,close)) / (high - low + 1e-9)
    'wick_ratio_down',# (min(open,close) - low) / (high - low + 1e-9)
    'volume_ratio_5', # volume / rolling_mean(volume, 5)
    'volume_ratio_20',# volume / rolling_mean(volume, 20)
    'vwap_dev',       # (close - vwap) / vwap, where vwap = sum(close*vol)/sum(vol) over 20 bars
    'autocorr_5',     # Autocorrelation of ret_1 at lag=5, rolling window=20
    'skew_20',        # Rolling skewness of ret_1, window=20
    'kurt_20',        # Rolling kurtosis of ret_1, window=20
    'realized_vol_ratio', # vol_5 / vol_60
    'trend_strength', # abs(ret_12) / vol_20
    'close_rank_48',  # Percentile rank of close among last 48 closes [0..1]
    'gap_open',       # (open_t - close_{t-1}) / close_{t-1}
    'overnight_ret',  # Same as gap_open (inter-bar jump; keep for compatibility)

    # --- C. Macro & Market Structure ---
    'funding_rate',   # Current 8-hour perpetual funding rate
    'funding_8h_ma',  # 8-bar rolling mean of funding_rate (=4h MA at 15-min bars)
    
    # --- D. Time & Session (cyclical) ---
    'hour_sin',       # sin(2π * hour / 24)
    'hour_cos',       # cos(2π * hour / 24)
    'dow_sin',        # sin(2π * day_of_week / 7)
    'dow_cos',        # cos(2π * day_of_week / 7)
    'is_asia_session',# 1 if hour in [0..8] UTC, else 0  (binary, NOT scaled)
    'is_us_session',  # 1 if hour in [13..21] UTC, else 0 (binary, NOT scaled)
    'is_weekend',     # 1 if day_of_week >= 5, else 0     (binary, NOT scaled)
    'minutes_to_funding', # Minutes until next 8h funding: 00:00, 08:00, 16:00 UTC
]
```

### 3.1 OBI Calculation

```python
import numpy as np

def calc_obi(bid_vols: np.ndarray, ask_vols: np.ndarray, tau: float) -> float:
    """
    Exponentially-decayed Order Book Imbalance across top N levels.
    bid_vols, ask_vols: arrays of length N (level 0 = best)
    tau: decay constant
    """
    N = len(bid_vols)
    levels = np.arange(N)
    weights = np.exp(-levels / tau)
    bid_w = np.sum(weights * bid_vols)
    ask_w = np.sum(weights * ask_vols)
    denom = bid_w + ask_w
    if denom == 0:
        return 0.0
    return (bid_w - ask_w) / denom
```

Call with `tau` in `{1, 3, 5, 10}` to get `obi_tau1` through `obi_tau10`.

### 3.2 Kyle's Lambda

```python
def calc_kyle_lambda(prices: pd.Series, volumes: pd.Series, window: int = 20) -> pd.Series:
    """
    Price impact proxy: |delta_price| / |delta_volume|.
    Rolling over `window` bars, averaged.
    """
    dp = prices.diff().abs()
    dv = volumes.diff().abs().replace(0, np.nan)
    return (dp / dv).rolling(window).mean().fillna(0)
```

### 3.3 RSI

```python
def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50) / 100   # Normalize to [0,1]
```

### 3.4 MACD Signal

```python
def calc_macd_signal(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    return macd.ewm(span=9, adjust=False).mean()
```

### 3.5 Bollinger Band % Position

```python
def calc_bb_pct(close: pd.Series, window: int = 20) -> pd.Series:
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return ((close - lower) / (upper - lower + 1e-9)).clip(0, 1)
```

### 3.6 VWAP Deviation

```python
def calc_vwap_dev(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    vwap = (close * volume).rolling(window).sum() / volume.rolling(window).sum()
    return (close - vwap) / (vwap + 1e-9)
```

### 3.7 Minutes to Funding

```python
def minutes_to_funding(ts: pd.DatetimeIndex) -> pd.Series:
    """Next funding event at 00:00, 08:00, 16:00 UTC."""
    funding_hours = [0, 8, 16]
    result = []
    for t in ts:
        minutes_since_midnight = t.hour * 60 + t.minute
        candidates = [h * 60 - minutes_since_midnight for h in funding_hours]
        candidates += [h * 60 + 1440 - minutes_since_midnight for h in funding_hours]
        result.append(min(m for m in candidates if m > 0))
    return pd.Series(result, index=ts)
```

### 3.8 Full Feature Builder

```python
# scripts/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

BINARY_FEATURES = {'is_asia_session', 'is_us_session', 'is_weekend'}

def build_features(df: pd.DataFrame, scaler=None, fit_scaler=False) -> pd.DataFrame:
    """
    df must have columns:
        open, high, low, close, volume,       (OHLCV)
        bid_vol_L{0..19}, ask_vol_L{0..19},   (L2 top 20 levels, volume)
        bid_price_L0, ask_price_L0,            (best bid/ask)
        funding_rate                            (current 8h rate)
    Index must be a UTC DatetimeIndex at 15-minute frequency.

    Returns: DataFrame with FEATURE_NAMES columns, NaN rows dropped.
    """
    out = pd.DataFrame(index=df.index)

    # --- Orderbook Microstructure ---
    bid_vols = df[[f'bid_vol_L{i}' for i in range(20)]].values
    ask_vols = df[[f'ask_vol_L{i}' for i in range(20)]].values

    for tau in [1, 3, 5, 10]:
        obi = []
        for b, a in zip(bid_vols, ask_vols):
            obi.append(calc_obi(b, a, tau))
        out[f'obi_tau{tau}'] = obi

    mid = (df['bid_price_L0'] + df['ask_price_L0']) / 2
    out['spread_bps']     = (df['ask_price_L0'] - df['bid_price_L0']) / mid * 10_000
    out['depth_ratio_5']  = df[[f'bid_vol_L{i}' for i in range(1,5)]].sum(axis=1) / \
                             df[[f'ask_vol_L{i}' for i in range(1,5)]].sum(axis=1).replace(0, np.nan)
    out['depth_ratio_10'] = df[[f'bid_vol_L{i}' for i in range(1,10)]].sum(axis=1) / \
                             df[[f'ask_vol_L{i}' for i in range(1,10)]].sum(axis=1).replace(0, np.nan)
    out['mid_price_move'] = mid.pct_change()
    out['book_pressure_3']= df[[f'bid_vol_L{i}' for i in range(2,4)]].sum(axis=1) / \
                             df[[f'ask_vol_L{i}' for i in range(2,4)]].sum(axis=1).replace(0, np.nan)
    out['kyle_lambda_est']= calc_kyle_lambda(df['close'], df['volume'])

    # --- Price Action ---
    log_ret = np.log(df['close'] / df['close'].shift(1))
    for lag, name in [(1,'ret_1'),(3,'ret_3'),(6,'ret_6'),(12,'ret_12'),(48,'ret_48')]:
        out[name] = np.log(df['close'] / df['close'].shift(lag))
    out['vol_5']  = log_ret.rolling(5).std()
    out['vol_20'] = log_ret.rolling(20).std()
    out['vol_60'] = log_ret.rolling(60).std()
    out['rsi_14'] = calc_rsi(df['close'], 14)
    out['rsi_6']  = calc_rsi(df['close'], 6)
    out['macd_signal'] = calc_macd_signal(df['close'])
    out['bb_pct'] = calc_bb_pct(df['close'])
    out['atr_14'] = (df['high'] - df['low']).rolling(14).mean() / df['close']
    bar_range = (df['high'] - df['low']).replace(0, 1e-9)
    out['momentum_bar']   = (df['close'] - df['open']) / bar_range
    out['wick_ratio_up']  = (df['high'] - df[['open','close']].max(axis=1)) / bar_range
    out['wick_ratio_down']= (df[['open','close']].min(axis=1) - df['low']) / bar_range
    out['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean().replace(0, np.nan)
    out['volume_ratio_20']= df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)
    out['vwap_dev']   = calc_vwap_dev(df['close'], df['volume'])
    out['autocorr_5'] = log_ret.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
    out['skew_20']    = log_ret.rolling(20).skew()
    out['kurt_20']    = log_ret.rolling(20).kurt()
    out['realized_vol_ratio'] = out['vol_5'] / out['vol_60'].replace(0, np.nan)
    out['trend_strength']     = out['ret_12'].abs() / out['vol_20'].replace(0, np.nan)
    out['close_rank_48'] = df['close'].rolling(48).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    out['gap_open']      = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    out['overnight_ret'] = out['gap_open']   # same signal; kept for model compatibility

    # --- Macro ---
    out['funding_rate']  = df['funding_rate']
    out['funding_8h_ma'] = df['funding_rate'].rolling(8).mean()

    # --- Time ---
    hour = df.index.hour
    dow  = df.index.dayofweek
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['dow_sin']  = np.sin(2 * np.pi * dow / 7)
    out['dow_cos']  = np.cos(2 * np.pi * dow / 7)
    out['is_asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    out['is_us_session']   = ((hour >= 13) & (hour < 21)).astype(int)
    out['is_weekend']      = (dow >= 5).astype(int)
    out['minutes_to_funding'] = minutes_to_funding(df.index)

    # Drop NaN rows (warm-up period)
    out = out.dropna()

    # Ensure column order matches model exactly
    out = out[FEATURE_NAMES]

    # Scale non-binary features
    non_binary = [c for c in FEATURE_NAMES if c not in BINARY_FEATURES]
    if fit_scaler:
        scaler = RobustScaler()
        out[non_binary] = scaler.fit_transform(out[non_binary])
        joblib.dump(scaler, 'models/feature_scaler.pkl')
    elif scaler is not None:
        out[non_binary] = scaler.transform(out[non_binary])

    return out, scaler
```

---

## 4. Model Inference (`scripts/backtester.py` — inference section)

```python
import xgboost as xgb
import numpy as np

CALIBRATION_FACTOR = 1.0411
BPS_TO_RATIO       = 1 / 10_000   # convert BPS prediction back to ratio delta

def load_model(model_path: str = 'models/xgb_regression_v0.json') -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def predict_ratios(model: xgb.Booster, features_df) -> np.ndarray:
    """
    Returns calibrated predicted ratio (Price_{t+1} / Price_t) for each bar.
    """
    dmatrix = xgb.DMatrix(features_df[FEATURE_NAMES])
    bps_raw = model.predict(dmatrix)                    # raw BPS prediction
    bps_cal = bps_raw * CALIBRATION_FACTOR              # apply variance scaling
    ratios  = 1.0 + (bps_cal * BPS_TO_RATIO)           # convert to price ratio
    return ratios
```

---

## 5. Trading Strategy Logic

The strategy converts calibrated ratio predictions into **discrete trading signals** and manages a position with risk controls.

### 5.1 Signal Generation

```python
# Tunable thresholds (start with these, then optimize on validation set)
LONG_THRESHOLD   = 1.000080   # predicted ratio > this → go LONG
SHORT_THRESHOLD  = 0.999920   # predicted ratio < this → go SHORT
# Dead zone: [0.999920, 1.000080] → FLAT (no position)

def generate_signal(predicted_ratio: float) -> int:
    """
    Returns: +1 (long), -1 (short), 0 (flat)
    """
    if predicted_ratio > LONG_THRESHOLD:
        return 1
    elif predicted_ratio < SHORT_THRESHOLD:
        return -1
    else:
        return 0
```

**Threshold calibration note**: The model's test-set predicted ratio std is `0.001053` (~10.5 BPS). Set thresholds at ±0.5 to ±1.0 sigma of predicted std, e.g.:

| Aggressiveness | Long Threshold | Short Threshold |
|---|---|---|
| Conservative (1σ) | 1.001053 | 0.998947 |
| Moderate (0.5σ) | 1.000527 | 0.999473 |
| **Default (0.08 BPS)** | **1.000080** | **0.999920** |

### 5.2 Position Sizing

```python
POSITION_SIZE_USD  = 10_000   # Fixed notional per trade in USD
MAX_LEVERAGE       = 3.0      # Never exceed 3x leverage on account equity
MAKER_FEE          = 0.0002   # 2 BPS (limit orders, adjust per exchange)
TAKER_FEE          = 0.0005   # 5 BPS (market orders)

def compute_position_size(equity: float, price: float) -> float:
    """Returns BTC quantity for this bar's trade."""
    max_notional = equity * MAX_LEVERAGE
    notional = min(POSITION_SIZE_USD, max_notional)
    return notional / price
```

### 5.3 Risk Management Rules

Apply these filters **before** entering a position:

| Rule | Condition | Action |
|---|---|---|
| High spread filter | `spread_bps > 10` | Force FLAT — cost too high |
| Volatility circuit breaker | `vol_5 > 3 * vol_60` | Force FLAT — regime instability |
| Funding rate filter | `abs(funding_rate) > 0.001` | Skip SHORT if rate > 0 (longs pay), skip LONG if rate < 0 |
| Weekend filter | `is_weekend == 1` | Optionally reduce position to 50% |
| Session override | `is_asia_session == 1` | Optionally reduce to 50% (lower liquidity) |

```python
def apply_risk_filters(signal: int, features_row: dict) -> int:
    """Returns filtered signal (may reduce to 0)."""
    # Spread too wide
    if features_row['spread_bps'] > 10:
        return 0
    # Volatility regime break
    if features_row['vol_5'] > 3 * features_row['vol_60']:
        return 0
    # Funding rate bias suppression
    fr = features_row['funding_rate']
    if signal == 1 and fr < -0.001:   # negative funding = bearish bias
        return 0
    if signal == -1 and fr > 0.001:   # positive funding = bullish bias
        return 0
    return signal
```

---

## 6. Backtester Engine (`scripts/backtester.py`)

```python
# scripts/backtester.py

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Trade:
    entry_bar:  int
    exit_bar:   int
    direction:  int          # +1 long, -1 short
    entry_price: float
    exit_price:  float
    qty_btc:     float
    pnl_usd:     float
    fees_usd:    float
    net_pnl_usd: float

@dataclass
class BacktestState:
    equity:       float = 100_000.0
    position:     int   = 0         # current signal: +1, -1, 0
    entry_price:  float = 0.0
    entry_bar:    int   = 0
    qty_btc:      float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades:       List[Trade]  = field(default_factory=list)


def run_backtest(
    ohlcv_df:    pd.DataFrame,   # original OHLCV (not feature-engineered), aligned to features_df
    features_df: pd.DataFrame,   # output of build_features()
    model,                       # loaded xgb.Booster
    initial_equity: float = 100_000.0,
) -> BacktestState:

    state = BacktestState(equity=initial_equity)
    predicted_ratios = predict_ratios(model, features_df)

    for i, (ts, feat_row) in enumerate(features_df.iterrows()):
        price_now  = ohlcv_df.loc[ts, 'close']
        pred_ratio = predicted_ratios[i]

        # --- Generate and filter signal ---
        raw_signal = generate_signal(pred_ratio)
        signal     = apply_risk_filters(raw_signal, feat_row.to_dict())

        # --- Close existing position if signal flips or goes flat ---
        if state.position != 0 and signal != state.position:
            exit_price = price_now
            pnl = state.qty_btc * (exit_price - state.entry_price) * state.position
            fees = (state.qty_btc * state.entry_price + state.qty_btc * exit_price) * TAKER_FEE
            net = pnl - fees
            state.equity += net
            state.trades.append(Trade(
                entry_bar=state.entry_bar, exit_bar=i,
                direction=state.position,
                entry_price=state.entry_price, exit_price=exit_price,
                qty_btc=state.qty_btc,
                pnl_usd=pnl, fees_usd=fees, net_pnl_usd=net
            ))
            state.position = 0
            state.qty_btc  = 0.0

        # --- Open new position ---
        if signal != 0 and state.position == 0:
            state.position   = signal
            state.entry_price = price_now
            state.entry_bar   = i
            state.qty_btc    = compute_position_size(state.equity, price_now)

        # --- Mark-to-market equity for equity curve ---
        if state.position != 0:
            unrealized = state.qty_btc * (price_now - state.entry_price) * state.position
            mtm_equity = state.equity + unrealized
        else:
            mtm_equity = state.equity
        state.equity_curve.append(mtm_equity)

    # Close any open position at end
    if state.position != 0:
        last_price = ohlcv_df['close'].iloc[-1]
        pnl  = state.qty_btc * (last_price - state.entry_price) * state.position
        fees = (state.qty_btc * state.entry_price + state.qty_btc * last_price) * TAKER_FEE
        net  = pnl - fees
        state.equity += net
        state.trades.append(Trade(
            entry_bar=state.entry_bar, exit_bar=len(features_df)-1,
            direction=state.position,
            entry_price=state.entry_price, exit_price=last_price,
            qty_btc=state.qty_btc,
            pnl_usd=pnl, fees_usd=fees, net_pnl_usd=net
        ))

    return state
```

---

## 7. Performance Metrics (`scripts/report.py`)

```python
import numpy as np
import pandas as pd

def compute_metrics(state, risk_free_rate_annual=0.05) -> dict:
    trades_df = pd.DataFrame([vars(t) for t in state.trades])
    equity    = np.array(state.equity_curve)
    returns   = np.diff(equity) / equity[:-1]   # bar-by-bar returns

    # --- Core metrics ---
    total_pnl     = trades_df['net_pnl_usd'].sum() if len(trades_df) else 0
    total_trades  = len(trades_df)
    win_trades    = (trades_df['net_pnl_usd'] > 0).sum() if len(trades_df) else 0
    win_rate      = win_trades / total_trades if total_trades > 0 else 0

    avg_win  = trades_df.loc[trades_df['net_pnl_usd'] > 0, 'net_pnl_usd'].mean() if win_trades else 0
    avg_loss = trades_df.loc[trades_df['net_pnl_usd'] < 0, 'net_pnl_usd'].mean() if (total_trades - win_trades) else 0
    profit_factor = abs(avg_win * win_trades) / abs(avg_loss * (total_trades - win_trades) + 1e-9)

    # --- Risk metrics ---
    bars_per_year  = 365 * 24 * 4   # 15-min bars in a year
    rf_per_bar     = (1 + risk_free_rate_annual) ** (1 / bars_per_year) - 1
    excess_returns = returns - rf_per_bar
    sharpe         = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(bars_per_year)

    rolling_max    = np.maximum.accumulate(equity)
    drawdowns      = (equity - rolling_max) / rolling_max
    max_drawdown   = drawdowns.min()
    calmar         = (total_pnl / 100_000) / abs(max_drawdown + 1e-9)

    # --- Sortino (downside deviation only) ---
    downside = excess_returns[excess_returns < 0]
    sortino  = np.mean(excess_returns) / (np.std(downside) + 1e-9) * np.sqrt(bars_per_year)

    # --- Total fees paid ---
    total_fees = trades_df['fees_usd'].sum() if len(trades_df) else 0

    return {
        'Total Net PnL ($)':     round(total_pnl, 2),
        'Total Trades':          total_trades,
        'Win Rate (%)':          round(win_rate * 100, 2),
        'Profit Factor':         round(profit_factor, 3),
        'Sharpe Ratio (annual)': round(sharpe, 3),
        'Sortino Ratio':         round(sortino, 3),
        'Max Drawdown (%)':      round(max_drawdown * 100, 2),
        'Calmar Ratio':          round(calmar, 3),
        'Avg Win ($)':           round(avg_win, 2),
        'Avg Loss ($)':          round(avg_loss, 2),
        'Total Fees Paid ($)':   round(total_fees, 2),
    }
```

---

## 8. Main Entry Point (`run_backtest.py`)

```python
# run_backtest.py
"""
Usage:
    python run_backtest.py --data data/raw/btc_15m.csv --mode oos
    python run_backtest.py --data data/raw/btc_15m.csv --mode full

CSV format required:
    timestamp (UTC ISO8601 or Unix ms), open, high, low, close, volume,
    bid_vol_L0..L19, ask_vol_L0..L19, bid_price_L0, ask_price_L0, funding_rate
"""

import argparse
import pandas as pd
import joblib
import os
from scripts.feature_engineering import build_features
from scripts.backtester import load_model, run_backtest
from scripts.report import compute_metrics

def main(data_path: str, mode: str):
    # --- Load raw data ---
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    df = df.sort_index()

    # --- Train/OOS split (80/20) ---
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    oos_df    = df.iloc[split_idx:]

    target_df = oos_df if mode == 'oos' else df

    # --- Feature engineering ---
    scaler_path = 'models/feature_scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        features, _ = build_features(target_df, scaler=scaler, fit_scaler=False)
    else:
        print("No saved scaler found. Fitting on training portion...")
        _, scaler = build_features(train_df, fit_scaler=True)
        features, _ = build_features(target_df, scaler=scaler, fit_scaler=False)

    # Align OHLCV to feature index (NaN warm-up rows dropped)
    ohlcv = target_df.loc[features.index]

    # --- Load model & run ---
    model  = load_model('models/xgb_regression_v0.json')
    state  = run_backtest(ohlcv, features, model)
    metrics = compute_metrics(state)

    # --- Report ---
    print("\n=== BACKTEST RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    trades_df = pd.DataFrame([vars(t) for t in state.trades])
    trades_df.to_csv('results/trades.csv', index=False)
    pd.DataFrame({'equity': state.equity_curve}).to_csv('results/equity_curve.csv', index=False)
    print("\nOutputs saved to results/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--mode', choices=['oos', 'full'], default='oos')
    args = parser.parse_args()
    main(args.data, args.mode)
```

---

## 9. Data Format Specification

Your input CSV (`data/raw/btc_15m.csv`) must contain these columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO8601 UTC | Bar open timestamp |
| `open`, `high`, `low`, `close` | float | OHLC prices in USD |
| `volume` | float | Base asset (BTC) volume for the bar |
| `bid_price_L0` | float | Best bid price |
| `ask_price_L0` | float | Best ask price |
| `bid_vol_L{0..19}` | float | Bid volume at each of 20 levels |
| `ask_vol_L{0..19}` | float | Ask volume at each of 20 levels |
| `funding_rate` | float | Current perpetual funding rate (e.g. `0.0001`) |

**If you only have OHLCV** (no L2 data), use these approximations at the cost of reduced accuracy:
- `bid_price_L0 = close * 0.9999`, `ask_price_L0 = close * 1.0001`
- All `bid_vol_L{i}` and `ask_vol_L{i}` = `volume * exp(-i * 0.5)` (synthetic decay)
- `spread_bps` will be pinned to a constant; OBI features will be approximate
- `kyle_lambda_est`, `depth_ratio_*`, `book_pressure_3` will be approximate

---

## 10. Implementation Checklist for Gemini CLI

Work through these tasks **in order**. Verify each step before proceeding.

- [ ] **Step 1**: Create project structure exactly as shown in Section 1
- [ ] **Step 2**: Install dependencies from `requirements.txt`
- [ ] **Step 3**: Copy `xgb_regression_v0.json` → `models/`
- [ ] **Step 4**: Implement `scripts/feature_engineering.py` — verify the 47 feature names match `FEATURE_NAMES` exactly and in order
- [ ] **Step 5**: Load a small sample (100 bars) and call `build_features()`. Print `features.columns.tolist()` and confirm it matches `FEATURE_NAMES`
- [ ] **Step 6**: Call `predict_ratios()` on those 100 bars. Predicted ratios should cluster near `1.000` (i.e., ±0.005); if not, check feature scaling
- [ ] **Step 7**: Implement `scripts/backtester.py` — run on the 100-bar sample and verify `state.trades` is populated without errors
- [ ] **Step 8**: Implement `scripts/report.py`
- [ ] **Step 9**: Run full backtest: `python run_backtest.py --data data/raw/btc_15m.csv --mode oos`
- [ ] **Step 10**: Validate: target **Sharpe > 1.5** and **Max Drawdown < 15%** on OOS; if below, tighten thresholds (increase `LONG_THRESHOLD`, decrease `SHORT_THRESHOLD`)

---

## 11. Common Pitfalls & Debugging

| Symptom | Likely Cause | Fix |
|---|---|---|
| All predictions ≈ 1.0 (no signal) | Scaler not fitted / wrong features | Re-check feature order; re-fit scaler on training split only |
| `DMatrix` shape error | Wrong number of features passed | Assert `features.shape[1] == 47` before predict |
| Very negative Sharpe | Thresholds too loose / fees too high | Widen thresholds; switch to maker fees |
| `NaN` in equity curve | Division by zero in feature calc | Add `.fillna(0)` after `build_features()` |
| OBI = 0 for all bars | L2 columns missing or named wrong | Verify column names match `bid_vol_L{i}` pattern |
| Funding rate always 0 | Missing funding data | Pull from exchange API (Binance, Bybit, etc.) at 8h intervals, forward-fill |
| Huge spike at bar 60+ | Warm-up NaN drop misaligned | Ensure `ohlcv = target_df.loc[features.index]` after feature build |

---

## 12. Extending the Backtester

Once baseline is working, consider these extensions:

**Stop-Loss / Take-Profit per Trade**
```python
STOP_LOSS_BPS   = -30    # Close if trade PnL < -30 BPS
TAKE_PROFIT_BPS = +50    # Close if trade PnL > +50 BPS

def check_exit_triggers(state, current_price):
    if state.position == 0:
        return False
    pnl_bps = (current_price - state.entry_price) / state.entry_price * 10_000 * state.position
    return pnl_bps < STOP_LOSS_BPS or pnl_bps > TAKE_PROFIT_BPS
```

**Rolling Walk-Forward Validation**  
Split data into 10 folds. For each fold: fit scaler on train, backtest on test. Average metrics across folds to avoid overfitting the threshold parameters.

**Slippage Model**  
Use `spread_bps` as a proxy for slippage. Adjust entry/exit price:
```python
slippage_ratio = features_row['spread_bps'] / 2 / 10_000
effective_entry = entry_price * (1 + signal * slippage_ratio)
```

**Live Paper Trading**  
Replace the CSV loop with a WebSocket listener (e.g., Binance futures stream). Call `build_features()` on a rolling buffer of the last 100 bars, run `predict_ratios()`, and submit orders via REST API.

---

*Report generated by ShinkáEvolve Model Registry | Model: xgb_regression_v0 | 968 trees | 47 features | Calibration: 1.0411*

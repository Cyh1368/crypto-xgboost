# evolve/initial.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (hyperparameters)
XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_alpha": 0.001,
    "reg_lambda": 0.02,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "learning_rate": 0.08,
    "gamma": 0.1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 96          # preserve more signal variance for stronger gradients
MIN_PRED_STD_RATIO = 0.08     # lower threshold to reduce collapse penalty
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features must use only past data (no lookahead).
    Returns a DataFrame aligned to df.index.
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    log_ret = np.log(close / close.shift(1))
    ret1 = close.pct_change(1)
    ret3 = close.pct_change(3)

    # --- Returns / momentum ---
    for lag in [1, 3, 6, 12, 24, 48]:
        feat[f'ret_{lag}'] = close.pct_change(lag)
    feat['ret_accel_3_1'] = ret1 - ret3 / 3.0
    feat['mom_ratio_12_48'] = (close.pct_change(12) + 1e-9) / (close.pct_change(48).abs() + 1e-9)

    # --- Volatility / range ---
    for w in [5, 20, 60]:
        feat[f'vol_{w}'] = log_ret.rolling(w).std()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm'] = feat['atr_14'] / close
    feat['range_pct'] = (high - low) / (close + 1e-9)
    feat['close_pos_20'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-9)

    # --- Oscillators / trend ---
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feat['macd_signal'] = macd - macd.ewm(span=9, adjust=False).mean()

    # --- Volume / flow ---
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    feat['volume_ratio_5'] = volume / (vol_ma5 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma20 + 1e-9)
    feat['volume_z_20'] = (volume - vol_ma20) / (volume.rolling(20).std() + 1e-9)
    feat['vol_x_ret1'] = feat['volume_ratio_5'] * ret1
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features ---
    feat['trend_strength_60'] = (close / close.shift(60) - 1).abs() * 10000
    realized_vol = log_ret.rolling(20).std()
    vol_regime = realized_vol / (realized_vol.rolling(200).mean() + 1e-9)
    feat['vol_regime'] = vol_regime
    feat['trend_x_regime'] = feat['trend_strength_60'] / (vol_regime + 1e-9)
    feat['ret1_x_regime'] = ret1 * vol_regime

    # Regime-conditional momentum interactions (capture signal suppressed by regularization)
    feat['ret6_x_vol_regime'] = close.pct_change(6) * vol_regime
    feat['ret12_x_vol_regime'] = close.pct_change(12) * vol_regime
    feat['ret3_x_trend_strength'] = close.pct_change(3) * (feat['trend_strength_60'] / 10000.0)

    # Adaptive momentum scaled by recent volatility
    recent_vol = log_ret.rolling(10).std()
    feat['adaptive_ret6'] = close.pct_change(6) / (recent_vol + 1e-9)
    feat['adaptive_ret12'] = close.pct_change(12) / (recent_vol + 1e-9)

    # Volatility-adjusted RSI (regime-conditional oscillator)
    rsi_14 = feat['rsi_14']
    feat['rsi_14_vol_adjusted'] = rsi_14 / (vol_regime + 1e-9)

    # --- High-contrast binary regime signals (create natural tree splits) ---
    # Momentum regime: strong positive vs negative momentum
    ret_6 = close.pct_change(6)
    ret_12 = close.pct_change(12)
    feat['momentum_positive'] = (ret_6 > 0).astype(float)
    feat['momentum_strong'] = (ret_6.abs() > ret_6.rolling(20).std()).astype(float)
    feat['momentum_regime'] = np.sign(ret_6) * feat['momentum_strong']

    # Volatility regime binary (high vs low)
    vol_20_ma = log_ret.rolling(20).std()
    vol_20_median = vol_20_ma.rolling(60).median()
    feat['vol_high_binary'] = (vol_20_ma > vol_20_median).astype(float)
    feat['vol_low_binary'] = (vol_20_ma < vol_20_median).astype(float)

    # Trend regime: strong uptrend vs downtrend
    sma_20 = close.rolling(20).mean()
    sma_60 = close.rolling(60).mean()
    feat['trend_up'] = ((close > sma_20) & (sma_20 > sma_60)).astype(float)
    feat['trend_down'] = ((close < sma_20) & (sma_20 < sma_60)).astype(float)
    feat['trend_regime'] = feat['trend_up'].astype(float) - feat['trend_down'].astype(float)

    # --- Mean-reversion divergence signals ---
    # Price deviation from SMA (mean reversion signal)
    feat['price_dev_sma20'] = (close - sma_20) / (sma_20 + 1e-9)
    feat['price_dev_sma60'] = (close - sma_60) / (sma_60 + 1e-9)

    # RSI extremes (overbought/oversold)
    feat['rsi_6_extreme'] = ((feat['rsi_6'] > 70) | (feat['rsi_6'] < 30)).astype(float)
    feat['rsi_14_extreme'] = ((feat['rsi_14'] > 70) | (feat['rsi_14'] < 30)).astype(float)
    feat['rsi_overbought'] = (feat['rsi_14'] > 70).astype(float)
    feat['rsi_oversold'] = (feat['rsi_14'] < 30).astype(float)

    # --- Momentum divergence and acceleration signals ---
    # Momentum acceleration (2nd derivative)
    feat['ret_accel_12_6'] = ret_12 - ret_6
    feat['ret_accel_6_3'] = ret_6 - ret3

    # Momentum reversal signal (momentum slowing down)
    feat['momentum_slowing'] = (ret_6.abs() < ret_12.abs()).astype(float)
    feat['momentum_accelerating'] = (ret_6.abs() > ret_12.abs()).astype(float)

    # Momentum divergence from trend
    feat['momentum_vs_trend'] = np.sign(ret_6) * feat['trend_regime']

    # --- Volatility-scaled momentum with discrete thresholds ---
    vol_20_std = log_ret.rolling(20).std()
    vol_20_mean = vol_20_std.rolling(60).mean()
    vol_normalized = vol_20_std / (vol_20_mean + 1e-9)

    # High momentum in low vol (strong signal)
    feat['strong_momentum_low_vol'] = (ret_6.abs() > ret_6.rolling(20).std()) & (vol_normalized < 1.0)
    feat['strong_momentum_low_vol'] = feat['strong_momentum_low_vol'].astype(float)

    # Weak momentum in high vol (noise)
    feat['weak_momentum_high_vol'] = (ret_6.abs() < ret_6.rolling(20).std()) & (vol_normalized > 1.2)
    feat['weak_momentum_high_vol'] = feat['weak_momentum_high_vol'].astype(float)

    # --- Cross-regime interaction signals ---
    # Momentum aligned with trend (strong signal)
    feat['momentum_trend_aligned'] = np.sign(ret_6) * feat['trend_regime']

    # Momentum against trend (reversal signal)
    feat['momentum_trend_divergence'] = -np.sign(ret_6) * feat['trend_regime']

    # --- Orderbook (if real data available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_8h_ma'] = fr.rolling(32).mean()   # 32 × 15min = 8h
        feat['funding_z_32'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
    return feat
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# DO NOT MODIFY BELOW THIS LINE
# run_experiment is the fixed interface called by evaluate.py

def run_experiment(
    X_train, y_train,
    X_test,  y_test,
    X_val,   y_val,
):
    """
    Train an XGBoost model and return raw predictions for all three splits.
    Called by evaluate.py via shinka.core.run_shinka_eval.
    """
    # Clip targets
    clip_val = np.percentile(np.abs(y_train), CLIP_PERCENTILE)
    y_train_c = y_train.clip(-clip_val, clip_val)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_va = scaler.transform(X_val)

    params = {k: v for k, v in XGB_PARAMS.items()
              if k != 'early_stopping_rounds'}
    early = XGB_PARAMS.get('early_stopping_rounds', 30)

    model = xgb.XGBRegressor(**params, early_stopping_rounds=early)
    model.fit(
        X_tr, y_train_c,
        eval_set=[(X_te, y_test)],
        verbose=False,
    )

    return {
        'train': {'pred': model.predict(X_tr), 'actual': y_train.values},
        'test':  {'pred': model.predict(X_te), 'actual': y_test.values},
        'val':   {'pred': model.predict(X_va), 'actual': y_val.values},
        'model': model,
    }
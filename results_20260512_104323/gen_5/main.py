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
    "n_estimators": 250,
    "max_depth": 4,
    "min_child_weight": 3,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.12,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 97
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with SHORT lookbacks and rolling z-score normalization.
    All features use only past data (no lookahead).
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    log_ret = np.log(close / close.shift(1))

    # --- Short-term returns (z-scored within 20-bar window) ---
    for lag in [1, 2, 3, 6, 12]:
        ret = close.pct_change(lag)
        ret_mean = ret.rolling(20).mean()
        ret_std = ret.rolling(20).std()
        feat[f'ret_{lag}_z'] = (ret - ret_mean) / (ret_std + 1e-9)

    # --- Volatility (short lookback, z-scored) ---
    for w in [3, 5, 10, 20]:
        vol = log_ret.rolling(w).std()
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        feat[f'vol_{w}_z'] = (vol - vol_mean) / (vol_std + 1e-9)

    # --- Volatility trend (rate of change) ---
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_trend'] = (vol_5 - vol_5.shift(5)) / (vol_5.shift(5) + 1e-9)
    feat['vol_accel'] = vol_5.diff(3)

    # --- RSI (short period, normalized) ---
    for period in [6, 10]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}'] = (rsi - 50) / 50  # normalize to [-1, 1]

    # --- Bollinger position (short window) ---
    bb_mid = close.rolling(10).mean()
    bb_std = close.rolling(10).std()
    feat['bb_pct_10'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # --- ATR (normalized by price) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    feat['atr_norm'] = atr / close

    # --- MACD (short periods) ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    macd = ema8 - ema17
    macd_signal = macd - macd.ewm(span=6, adjust=False).mean()
    feat['macd_signal_z'] = (macd_signal - macd_signal.rolling(20).mean()) / (macd_signal.rolling(20).std() + 1e-9)

    # --- Volume (z-scored) ---
    vol_ratio_5 = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_5_z'] = (vol_ratio_5 - vol_ratio_5.rolling(20).mean()) / (vol_ratio_5.rolling(20).std() + 1e-9)
    
    vol_ratio_10 = volume / (volume.rolling(10).mean() + 1e-9)
    feat['volume_ratio_10_z'] = (vol_ratio_10 - vol_ratio_10.rolling(20).mean()) / (vol_ratio_10.rolling(20).std() + 1e-9)

    # --- VWAP deviation (short window) ---
    vwap = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Time features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features (short-term, relative) ---
    trend_10 = (close / close.shift(10) - 1)
    trend_20 = (close / close.shift(20) - 1)
    feat['trend_10_z'] = (trend_10 - trend_10.rolling(20).mean()) / (trend_10.rolling(20).std() + 1e-9)
    feat['trend_20_z'] = (trend_20 - trend_20.rolling(20).mean()) / (trend_20.rolling(20).std() + 1e-9)
    
    # Volatility regime (relative to recent history only)
    realized_vol_10 = log_ret.rolling(10).std()
    realized_vol_20 = log_ret.rolling(20).std()
    feat['vol_regime'] = realized_vol_10 / (realized_vol_20 + 1e-9)

    # --- Mean reversion (short window) ---
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    feat['price_sma5_dev'] = (close - sma_5) / (sma_5 + 1e-9)
    feat['price_sma20_dev'] = (close - sma_20) / (sma_20 + 1e-9)
    feat['sma_cross'] = (sma_5 - sma_20) / (sma_20 + 1e-9)

    # --- Momentum × Volatility interaction (KEY: single multiplicative term) ---
    ret_3 = close.pct_change(3)
    vol_10 = log_ret.rolling(10).std()
    feat['momentum_vol_interaction'] = ret_3 * vol_10
    feat['momentum_regime'] = ret_3 / (vol_10 + 1e-9)
    feat['momentum_vol_regime'] = ret_3 * feat['vol_regime']

    # --- Volume × Price interaction ---
    feat['volume_price_pressure'] = vol_ratio_5 * feat['price_sma5_dev']

    # --- High-low range ---
    feat['hl_range'] = (high - low) / close
    feat['hl_range_ma'] = feat['hl_range'].rolling(5).mean()
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)

    # --- Close position in recent range ---
    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    feat['close_position_10'] = (close - low_10) / (high_10 - low_10 + 1e-9)

    # --- Return acceleration ---
    feat['ret_accel_3'] = close.pct_change(1).diff(3)

    # --- Autocorrelation (short window) ---
    ret_1 = close.pct_change(1)
    feat['autocorr_5'] = ret_1.rolling(5).apply(lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0, raw=False)
    
    # --- Autocorrelation trend (rate of change) ---
    autocorr_10 = ret_1.rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0, raw=False)
    feat['autocorr_trend'] = autocorr_10.diff(5)

    # --- Efficiency ratio (short window) ---
    efficiency_10 = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_10'] = efficiency_10
    feat['efficiency_trend'] = efficiency_10.diff(5)

    # --- Volume-return correlation (short window) ---
    feat['vol_ret_corr_10'] = log_ret.rolling(10).corr(volume.pct_change(1))

    # --- Orderbook (if available) ---
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

    # --- Funding rate (if available, short-term only) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_4h_ma'] = fr.rolling(16).mean()   # 16 × 15min = 4h
        feat['funding_momentum'] = fr.diff(2)
        feat['funding_x_volregime'] = fr * feat['vol_regime']

    # --- Price momentum in different regimes ---
    feat['momentum_high_vol'] = ret_3 * (feat['vol_regime'] > 1.0).astype(float)
    feat['momentum_low_vol'] = ret_3 * (feat['vol_regime'] <= 1.0).astype(float)

    # --- Trend consistency (short window) ---
    ret_signs = np.sign(close.diff())
    feat['trend_consistency_10'] = ret_signs.rolling(10).mean()

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
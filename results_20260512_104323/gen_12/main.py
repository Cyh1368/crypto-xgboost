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
    "n_estimators": 350,
    "max_depth": 5,
    "min_child_weight": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 0.6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.09,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 45,
}

CLIP_PERCENTILE = 98
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with SHORT lookbacks and ROLLING Z-SCORE normalization.
    Added features to measure the rate of change of market character.
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    log_ret = np.log(close / close.shift(1))

    # --- Short-term returns (z-scored within 40-bar window for stability) ---
    for lag in [1, 2, 3, 6, 12]:
        ret = close.pct_change(lag)
        ret_mean = ret.rolling(40).mean()
        ret_std = ret.rolling(40).std()
        feat[f'ret_{lag}_z'] = (ret - ret_mean) / (ret_std + 1e-9)

    # --- Short-term volatility (z-scored) ---
    for w in [5, 10, 20]:
        vol = log_ret.rolling(w).std()
        vol_mean = vol.rolling(40).mean()
        vol_std = vol.rolling(40).std()
        feat[f'vol_{w}_z'] = (vol - vol_mean) / (vol_std + 1e-9)

    # --- Volatility regime (rate of change and long-term ratio) ---
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    vol_120 = log_ret.rolling(120).std()
    feat['vol_regime_short'] = vol_5 / (vol_20 + 1e-9)
    feat['vol_regime_long'] = vol_20 / (vol_120 + 1e-9)
    feat['vol_trend'] = (vol_5 - vol_20) / (vol_20 + 1e-9)

    # Volatility of Volatility (VoV) - captures regime transition risk
    feat['vov_10'] = vol_20.rolling(10).std() / (vol_20.rolling(40).mean() + 1e-9)

    # --- RSI (z-scored) ---
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}_z'] = (rsi - 50) / 25.0

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct_20'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # --- ATR (normalized) ---
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat['atr_norm'] = atr / (close + 1e-9)

    # --- MACD ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd = ema8 - ema21
    macd_signal = macd - macd.ewm(span=9, adjust=False).mean()
    feat['macd_signal_z'] = (macd_signal - macd_signal.rolling(40).mean()) / (macd_signal.rolling(40).std() + 1e-9)

    # --- Volume dynamics ---
    vol_ratio_10 = volume / (volume.rolling(10).mean() + 1e-9)
    feat['volume_ratio_z'] = (vol_ratio_10 - vol_ratio_10.rolling(40).mean()) / (vol_ratio_10.rolling(40).std() + 1e-9)
    feat['volume_surge'] = vol_ratio_10

    # --- VWAP deviation ---
    vwap = (close * volume).rolling(15).sum() / (volume.rolling(15).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Time features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Trend features ---
    trend_10 = close.pct_change(10)
    feat['trend_strength_10'] = trend_10.abs()
    feat['trend_dir_10'] = np.sign(trend_10)

    # --- Mean reversion ---
    sma_10 = close.rolling(10).mean()
    feat['price_sma10_dev'] = (close - sma_10) / (sma_10 + 1e-9)

    # --- Interactions ---
    ret_5 = close.pct_change(5)
    feat['momentum_vol_interaction'] = ret_5 * feat['vol_regime_short']
    feat['momentum_regime'] = ret_5 / (vol_20 + 1e-9)
    feat['volume_price_pressure'] = feat['volume_ratio_z'] * feat['price_sma10_dev']

    # --- Range and Body ---
    hl_range = (high - low) / (close + 1e-9)
    feat['hl_range_z'] = (hl_range - hl_range.rolling(40).mean()) / (hl_range.rolling(40).std() + 1e-9)
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)

    # --- Efficiency ratio (Kaufman's ER) and its trend ---
    efficiency_10 = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    efficiency_40 = (close - close.shift(40)).abs() / (close.diff().abs().rolling(40).sum() + 1e-9)
    feat['efficiency_10'] = efficiency_10
    feat['efficiency_trend'] = efficiency_10 - efficiency_40

    # --- Autocorrelation (trend persistence) ---
    feat['autocorr_10'] = log_ret.rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0, raw=False)

    # --- Orderbook (if available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                bid_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_v - ask_v) / (bid_v + ask_v + 1e-9)
            except: return 0.0
        feat['obi_z'] = df.apply(lambda r: obi(r, 1), axis=1)

    # --- Funding rate (if available) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate_z'] = (fr - fr.rolling(40).mean()) / (fr.rolling(40).std() + 1e-9)
        feat['funding_momentum'] = fr.diff(4)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
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
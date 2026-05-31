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
    "n_estimators": 400,
    "max_depth": 5,
    "min_child_weight": 2,
    "reg_alpha": 0.05,
    "reg_lambda": 0.6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.15     # minimum acceptable pred_std / actual_std
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

    # --- Volatility-normalized returns (regime-invariant) ---
    vol_ref = log_ret.rolling(20).std() + 1e-9
    for lag in [1, 2, 3, 5, 10, 15]:
        feat[f'ret_n_{lag}'] = close.pct_change(lag) / (vol_ref * np.sqrt(lag))

    # --- Z-scored Volatility ---
    for w in [10, 20]:
        v = log_ret.rolling(w).std()
        feat[f'vol_z_{w}'] = (v - v.rolling(100).mean()) / (v.rolling(100).std() + 1e-9)
    feat['vol_rank_100'] = log_ret.rolling(20).std().rolling(100).rank(pct=True)

    # --- Z-scored RSI (regime-normalized) ---
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi_raw = 100 - 100 / (1 + rs)
        # Z-score RSI within 20-bar window
        rsi_mean = rsi_raw.rolling(20).mean()
        rsi_std = rsi_raw.rolling(20).std()
        feat[f'rsi_{period}_z'] = (rsi_raw - rsi_mean) / (rsi_std + 1e-9)

    # --- Bollinger position (already normalized) ---
    bb_mid = close.rolling(15).mean()
    bb_std = close.rolling(15).std()
    feat['bb_pct_15'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # --- ATR normalized ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_10 = tr.rolling(10).mean()
    feat['atr_norm_10'] = atr_10 / close

    # --- MACD (shorter periods) ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    macd = ema8 - ema17
    macd_signal = macd.ewm(span=6, adjust=False).mean()
    feat['macd_diff'] = (macd - macd_signal) / close

    # --- Volume ratios (shorter windows) ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_10'] = volume / (volume.rolling(10).mean() + 1e-9)

    # --- Z-scored volume ---
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat['volume_z_20'] = (volume - vol_mean_20) / (vol_std_20 + 1e-9)

    # --- VWAP deviation ---
    vwap_15 = (close * volume).rolling(15).sum() / (volume.rolling(15).sum() + 1e-9)
    feat['vwap_dev_15'] = (close - vwap_15) / close

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    # --- Volatility regime (shorter reference window) ---
    realized_vol_10 = log_ret.rolling(10).std()
    realized_vol_20 = log_ret.rolling(20).std()
    vol_mean_60 = realized_vol_20.rolling(60).mean()
    feat['vol_regime'] = realized_vol_20 / (vol_mean_60 + 1e-9)

    # --- Volatility trend (rate of change) ---
    feat['vol_trend'] = (realized_vol_10 - realized_vol_10.shift(5)) / (realized_vol_10.shift(5) + 1e-9)

    # --- Mean reversion (z-scored price position) ---
    sma_10 = close.rolling(10).mean()
    sma_std_10 = close.rolling(10).std()
    feat['price_z_10'] = (close - sma_10) / (sma_std_10 + 1e-9)

    sma_20 = close.rolling(20).mean()
    sma_std_20 = close.rolling(20).std()
    feat['price_z_20'] = (close - sma_20) / (sma_std_20 + 1e-9)

    # --- Momentum × Volatility regime (key interaction) ---
    ret_3 = close.pct_change(3)
    ret_5 = close.pct_change(5)
    feat['momentum_vol_regime'] = ret_3 * feat['vol_regime']
    feat['momentum_vol_trend'] = ret_5 * feat['vol_trend']

    # --- Return acceleration ---
    feat['ret_accel_3'] = ret_3.diff(3)
    feat['ret_accel_5'] = ret_5.diff(5)

    # --- High-low range (normalized) ---
    hl_range = (high - low) / close
    feat['hl_range'] = hl_range
    hl_mean = hl_range.rolling(10).mean()
    hl_std = hl_range.rolling(10).std()
    feat['hl_range_z'] = (hl_range - hl_mean) / (hl_std + 1e-9)

    # --- Candle patterns (normalized) ---
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)

    # --- Close position in recent range ---
    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    feat['close_pos_10'] = (close - low_10) / (high_10 - low_10 + 1e-9)

    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['close_pos_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # --- Robust Price Indicators ---
    feat['stoch_k_14'] = (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-9)

    # --- Efficiency ratio (shorter window) ---
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)

    # --- Volume-price pressure ---
    feat['vol_price_pressure'] = feat['volume_z_20'] * feat['price_z_10']

    # --- Trend consistency ---
    ret_signs = np.sign(close.diff())
    feat['trend_consistency_10'] = ret_signs.rolling(10).mean()

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

    # --- Funding rate (if available) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        fr_mean = fr.rolling(20).mean()
        fr_std = fr.rolling(20).std()
        feat['funding_z'] = (fr - fr_mean) / (fr_std + 1e-9)
        feat['funding_trend'] = fr.diff(4)

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
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
    "n_estimators": 1100,
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_alpha": 0.01,
    "reg_lambda": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "gamma": 0.01,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 100,
}

CLIP_PERCENTILE = 98.5
MIN_PRED_STD_RATIO = 0.15
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

    # Use OHLCV from the bar that just closed (standard for predicting next bar)
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    log_close = np.log(close)
    log_ret = log_close.diff()

    # --- Core returns / momentum ---
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    feat['ret_accel_3'] = feat['ret_1'] - feat['ret_3']
    feat['ret_accel_6'] = feat['ret_3'] - feat['ret_6']
    feat['ret_mom_ratio'] = feat['ret_3'] / (feat['ret_12'].abs() + 1e-9)

    # --- Multi-scale volatility ---
    for w in [5, 10, 20, 40, 80]:
        feat[f'vol_{w}'] = log_ret.rolling(w).std()

    feat['vol_ratio_5_20'] = feat['vol_5'] / (feat['vol_20'] + 1e-9)
    feat['vol_ratio_20_80'] = feat['vol_20'] / (feat['vol_80'] + 1e-9)

    rv20 = log_ret.rolling(20).std()
    rv80 = log_ret.rolling(80).std()
    feat['vol_regime_fast'] = rv20 / (rv80 + 1e-9)
    feat['vol_regime_long'] = rv80 / (rv80.rolling(240).mean() + 1e-9)

    # --- Trend / mean reversion ---
    for w in [10, 20, 50, 100]:
        sma = close.rolling(w).mean()
        feat[f'price_sma{w}_dev'] = (close - sma) / (sma + 1e-9)

    feat['sma_cross_10_50'] = (
        close.rolling(10).mean() - close.rolling(50).mean()
    ) / (close.rolling(50).mean() + 1e-9)

    feat['slope_20'] = (close / close.shift(20) - 1) / 20.0
    feat['slope_60'] = (close / close.shift(60) - 1) / 60.0
    feat['trend_strength_60'] = (close / close.shift(60) - 1).abs() * 10000

    # --- Oscillators ---
    delta = close.diff()
    gain_7 = delta.clip(lower=0).rolling(7).mean()
    loss_7 = (-delta.clip(upper=0)).rolling(7).mean()
    rs_7 = gain_7 / (loss_7 + 1e-9)
    feat['rsi_7'] = 100 - 100 / (1 + rs_7)

    gain_14 = delta.clip(lower=0).rolling(14).mean()
    loss_14 = (-delta.clip(upper=0)).rolling(14).mean()
    rs_14 = gain_14 / (loss_14 + 1e-9)
    feat['rsi_14'] = 100 - 100 / (1 + rs_14)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_z'] = (close - bb_mid) / (bb_std + 1e-9)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat['stoch_k_14'] = (close - low_14) / (high_14 - low_14 + 1e-9)

    # --- MACD / EMA structure ---
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_mid = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=55, adjust=False).mean()
    macd = ema_fast - ema_slow
    feat['macd_signal'] = macd - macd.ewm(span=9, adjust=False).mean()
    feat['ema_spread_fast_mid'] = (ema_fast - ema_mid) / (close + 1e-9)
    feat['ema_spread_mid_slow'] = (ema_mid - ema_slow) / (close + 1e-9)

    # --- Candle / range structure ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feat['atr_norm'] = atr_14 / (close + 1e-9)
    feat['hl_range_norm'] = (high - low) / (close + 1e-9)
    feat['hl_range_ma_10'] = feat['hl_range_norm'].rolling(10).mean()

    body = (close - open_)
    full_range = (high - low) + 1e-9
    feat['body_ratio'] = body.abs() / full_range
    feat['body_dir'] = np.sign(body)
    feat['upper_wick_ratio'] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / full_range
    feat['lower_wick_ratio'] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / full_range

    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['close_position_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # --- Volume / flow ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_ma_60 = volume.rolling(60).mean()
    feat['volume_ratio_5'] = volume / (vol_ma_5 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma_20 + 1e-9)
    feat['volume_ratio_60'] = volume / (vol_ma_60 + 1e-9)
    feat['volume_z_20'] = (volume - vol_ma_20) / (volume.rolling(20).std() + 1e-9)
    feat['vol_momentum_5'] = volume.pct_change(5)
    feat['dollar_volume_z_20'] = ((volume * close) - (volume * close).rolling(20).mean()) / ((volume * close).rolling(20).std() + 1e-9)

    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap_20) / (close + 1e-9)

    feat['price_volume_corr_20'] = log_ret.rolling(20).corr(volume.pct_change(1))
    feat['price_volume_corr_60'] = log_ret.rolling(60).corr(volume.pct_change(1))

    # --- Statistical shape ---
    feat['zscore_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    feat['zscore_60'] = (close - close.rolling(60).mean()) / (close.rolling(60).std() + 1e-9)
    feat['skew_20'] = log_ret.rolling(20).skew()
    feat['kurt_20'] = log_ret.rolling(20).kurt()
    feat['vol_of_vol_20'] = log_ret.rolling(20).std().rolling(20).std()

    # --- Regime interactions ---
    feat['mom_x_vol'] = feat['ret_3'] * feat['vol_20']
    feat['mom_x_regime'] = feat['ret_6'] / (feat['vol_20'] + 1e-9)
    feat['trend_x_volreg'] = feat['slope_20'] * feat['vol_regime_fast']
    feat['range_x_vol'] = feat['hl_range_norm'] * feat['vol_20']

    # --- Time / session ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

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
        feat['obi_spread'] = feat['obi_tau1'] - feat['obi_tau3']

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_8h_ma'] = fr.rolling(32).mean()
        feat['funding_velocity'] = fr.diff(4)
        feat['funding_z_32'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)

    # Additional signal features
    feat['vol_cv_20'] = close.rolling(20).std() / (close.rolling(20).mean() + 1e-9)
    feat['dist_h_100'] = (high.rolling(100).max() - close) / (close + 1e-9)
    feat['dist_l_100'] = (close - low.rolling(100).min()) / (close + 1e-9)
    feat['force_index_7'] = (close.pct_change(1) * (volume / (volume.rolling(20).mean() + 1e-9))).rolling(7).mean()

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
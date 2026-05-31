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
    "n_estimators": 320,
    "max_depth": 4,
    "min_child_weight": 4,
    "reg_alpha": 0.15,
    "reg_lambda": 1.2,
    "gamma": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 97          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
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
    open_ = df['open']
    volume = df['volume']
    high = df['high']
    low = df['low']

    log_ret = np.log(close / close.shift(1))
    ret_1 = close.pct_change(1)
    ret_3 = close.pct_change(3)
    ret_6 = close.pct_change(6)

    # --- Returns: shorter lookbacks + regime-invariant z-scores ---
    for lag in [1, 2, 3, 5, 8, 12, 24]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    for w in [5, 10, 20]:
        r_mu = ret_1.rolling(w).mean()
        r_sd = ret_1.rolling(w).std()
        feat[f'ret_z_{w}'] = (ret_1 - r_mu) / (r_sd + 1e-9)

    # --- Volatility: short windows + change in market character ---
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_5'] = vol_5
    feat['vol_10'] = vol_10
    feat['vol_20'] = vol_20
    feat['vol_regime'] = vol_20 / (vol_20.rolling(100).mean() + 1e-9)
    feat['vol_regime_fast'] = vol_10 / (vol_20 + 1e-9)
    feat['vol_trend'] = (vol_10 - vol_10.shift(5)) / (vol_10.shift(5) + 1e-9)
    feat['vol_accel'] = feat['vol_trend'].diff(3)

    # --- RSI / oscillators ---
    for period in [5, 10, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Bollinger / price position ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    feat['price_sma5_dev'] = (close - sma_5) / (sma_5 + 1e-9)
    feat['price_sma10_dev'] = (close - sma_10) / (sma_10 + 1e-9)
    feat['price_sma10_dev_z'] = (feat['price_sma10_dev'] - feat['price_sma10_dev'].rolling(20).mean()) / (feat['price_sma10_dev'].rolling(20).std() + 1e-9)
    feat['sma_cross'] = (sma_5 - sma_20) / (sma_20 + 1e-9)

    # --- ATR / candle structure ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm'] = feat['atr_14'] / close
    feat['hl_range'] = (high - low) / (close + 1e-9)
    feat['hl_range_ma'] = feat['hl_range'].rolling(8).mean()
    feat['candle_body_ratio'] = (close - open_).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Trend / efficiency / autocorrelation change ---
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat['efficiency_trend'] = feat['efficiency_10'] - feat['efficiency_20']
    feat['efficiency_trend_z'] = (feat['efficiency_trend'] - feat['efficiency_trend'].rolling(20).mean()) / (feat['efficiency_trend'].rolling(20).std() + 1e-9)
    feat['autocorr_10'] = log_ret.rolling(10).corr(log_ret.shift(1))
    feat['autocorr_20'] = log_ret.rolling(20).corr(log_ret.shift(1))
    feat['autocorr_trend'] = feat['autocorr_10'] - feat['autocorr_20']

    # --- Volume normalization ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    feat['volume_ratio_5'] = volume / (vol_ma_5 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma_20 + 1e-9)
    feat['volume_z_20'] = (volume - vol_ma_20) / (volume.rolling(20).std() + 1e-9)
    feat['volume_trend'] = (vol_ma_5 - vol_ma_20) / (vol_ma_20 + 1e-9)

    # --- Momentum × regime interactions ---
    feat['momentum_z_3'] = (ret_3 - ret_3.rolling(20).mean()) / (ret_3.rolling(20).std() + 1e-9)
    feat['momentum_z_6'] = (ret_6 - ret_6.rolling(20).mean()) / (ret_6.rolling(20).std() + 1e-9)
    feat['momentum_regime'] = ret_3 / (vol_20 + 1e-9)
    feat['momentum_vol_interaction'] = feat['momentum_z_3'] * feat['vol_regime']
    feat['momentum_regime_fast'] = ret_3 * feat['vol_regime_fast']
    feat['volume_price_pressure'] = feat['volume_ratio_20'] * feat['price_sma10_dev']
    feat['volume_bb_interaction'] = feat['volume_ratio_20'] * feat['bb_pct']

    # --- Time features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
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

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_8h_ma'] = fr.rolling(32).mean()
        feat['funding_momentum'] = fr.diff(4)
        feat['funding_x_volregime'] = fr * feat['vol_regime']
        feat['funding_x_momentum'] = fr * ret_3

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
        feat['funding_momentum'] = fr.diff(4)
        feat['funding_x_volregime'] = fr * feat['vol_regime']

    # --- Trend Consistency and Efficiency ---
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)

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
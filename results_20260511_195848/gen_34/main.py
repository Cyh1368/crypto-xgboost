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
    "n_estimators": 300,
    "max_depth": 6,              # increased depth for more expressivity
    "min_child_weight": 1,
    "reg_alpha": 0.0,            # removed L1 regularization to increase expressivity
    "reg_lambda": 0.5,           # reduce L2 regularization to loosen constraints
    "subsample": 1.0,            # use all data per tree to improve expressivity
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,        # increased learning rate for stronger fitting
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

CLIP_PERCENTILE = 99          # clip targets at this percentile (both tails)
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

    # --- Returns ---
    # Use shifted pct_change to avoid lookahead
    for lag in [1, 3, 6, 12, 48]:
        feat[f'ret_{lag}'] = close.pct_change(lag).shift(1)

    # Additional momentum feature (ret_1 - ret_3)
    feat['momentum_1_3'] = (feat['ret_1'] - feat['ret_3'])
    
    # Clip momentum feature to remove extreme outliers
    momentum_clip_val = np.percentile(np.abs(feat['momentum_1_3'].dropna()), 99)
    feat['momentum_1_3'] = feat['momentum_1_3'].clip(-momentum_clip_val, momentum_clip_val)

    # --- Volatility ---
    log_ret = np.log(close / close.shift(1))
    for w in [5, 20, 60]:
        feat[f'vol_{w}'] = log_ret.rolling(w).std().shift(1)

    # --- RSI ---
    delta = close.diff()
    for period in [6, 14]:
        gain = delta.clip(lower=0).rolling(period).mean().shift(1)
        loss = (-delta.clip(upper=0)).rolling(period).mean().shift(1)
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean().shift(1)
    bb_std = close.rolling(20).std().shift(1)
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean().shift(1)
    feat['atr_norm'] = feat['atr_14'] / close

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean().shift(1)
    ema26 = close.ewm(span=26, adjust=False).mean().shift(1)
    macd = ema12 - ema26
    macd_signal = macd - macd.ewm(span=9, adjust=False).mean()
    feat['macd_signal'] = macd_signal.shift(1)

    # --- Volume ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_5'] = feat['volume_ratio_5'].shift(1)
    feat['volume_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    feat['volume_ratio_20'] = feat['volume_ratio_20'].shift(1)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)
    feat['vwap_dev'] = feat['vwap_dev'].shift(1)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features ---
    # Use shifted values to avoid lookahead
    feat['trend_strength_20'] = (close / close.shift(20) - 1).abs().shift(1) * 10000
    feat['trend_strength_60'] = (close / close.shift(60) - 1).abs().shift(1) * 10000
    realized_vol = log_ret.rolling(20).std()
    feat['vol_regime'] = (realized_vol / (realized_vol.rolling(200).mean() + 1e-9)).shift(1)

    # Additional regime feature: realized vol ratio short/long
    vol_short = log_ret.rolling(10).std()
    vol_long = log_ret.rolling(100).std()
    feat['vol_ratio_10_100'] = (vol_short / (vol_long + 1e-9)).shift(1)

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
        # Compute OBI on shifted rows to avoid leakage
        feat['obi_tau1'] = df.shift(1).apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.shift(1).apply(lambda r: obi(r, 3), axis=1)

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].shift(1)
        feat['funding_rate'] = fr
        feat['funding_8h_ma'] = fr.rolling(32).mean()

    # Replace inf and dropna after shifting
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

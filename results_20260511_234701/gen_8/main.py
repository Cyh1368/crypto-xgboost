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
    "max_depth": 3,
    "min_child_weight": 12,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

CLIP_PERCENTILE = 99          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.20     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-invariant features using rolling z-scores and bounded technicals.
    """
    feat = pd.DataFrame(index=df.index)
    close, volume, high, low = df['close'], df['volume'], df['high'], df['low']
    log_ret = np.log(close / close.shift(1))

    # --- Rolling Z-Scored Returns (Regime-Invariant) ---
    for lag in [1, 3, 5, 12]:
        r = close.pct_change(lag)
        feat[f'ret_z_{lag}'] = (r - r.rolling(100).mean()) / (r.rolling(100).std() + 1e-9)

    # --- Volatility Dynamics ---
    for w in [10, 30]:
        v = log_ret.rolling(w).std()
        feat[f'vol_z_{w}'] = (v - v.rolling(100).mean()) / (v.rolling(100).std() + 1e-9)
    feat['vol_ratio'] = log_ret.rolling(5).std() / (log_ret.rolling(20).std() + 1e-9)

    # --- Kaufman Efficiency Ratio (Market Character) ---
    for w in [10, 30]:
        direction = (close - close.shift(w)).abs()
        volatility = close.diff().abs().rolling(w).sum()
        feat[f'efficiency_{w}'] = direction / (volatility + 1e-9)

    # --- Bounded Technicals ---
    for period in [14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        feat[f'rsi_{period}'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['vwap_dev'] = (close - (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)) / (close + 1e-9)

    # --- Volume Z-Score ---
    log_v = np.log(volume + 1)
    feat['volume_z'] = (log_v - log_v.rolling(100).mean()) / (log_v.rolling(100).std() + 1e-9)

    # --- Price Action / Wick Details ---
    feat['wick_upper'] = (high - df[['open', 'close']].max(axis=1)) / (high - low + 1e-9)
    feat['wick_lower'] = (df[['open', 'close']].min(axis=1) - low) / (high - low + 1e-9)
    feat['body_ratio'] = (close - df['open']).abs() / (high - low + 1e-9)

    # --- Relative Range ---
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_z'] = (tr - tr.rolling(100).mean()) / (tr.rolling(100).std() + 1e-9)

    # --- Interactions (Normalized) ---
    feat['mom_vol_z'] = feat['ret_z_3'] * feat['vol_z_10']
    feat['trend_strength_z'] = (close / close.shift(20) - 1).abs().rolling(100).rank(pct=True)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- External Data (if available) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_z'] = (fr - fr.rolling(100).mean()) / (fr.rolling(100).std() + 1e-9)

    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + 1e-9)
            except: return 0.0
        feat['obi'] = df.apply(lambda r: obi(r, 2), axis=1)

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
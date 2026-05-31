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
    "n_estimators": 500,
    "max_depth": 3,
    "min_child_weight": 30,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 99          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.20     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-invariant features using rolling z-scores and relative technicals.
    """
    feat = pd.DataFrame(index=df.index)
    close, volume, high, low, open_ = df['close'], df['volume'], df['high'], df['low'], df['open']
    eps = 1e-9

    def rz(x, w):
        return (x - x.rolling(w).mean()) / (x.rolling(w).std() + eps)

    # --- Normalized Returns ---
    log_ret = np.log(close / close.shift(1))
    for lag in [1, 3, 5, 8, 12]:
        r = close.pct_change(lag)
        feat[f'ret_z_{lag}'] = rz(r, 60)
        feat[f'ret_vola_{lag}'] = r / (log_ret.rolling(20).std() + eps)

    # --- Volatility Structure ---
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_z_20'] = rz(vol_20, 60)
    feat['vol_ratio'] = vol_5 / (vol_20 + eps)
    feat['vol_trend'] = vol_20 / (vol_20.rolling(100).mean() + eps)

    # --- RSI (Z-scored) ---
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rsi = 100 - 100 / (1 + gain / (loss + eps))
        feat[f'rsi_z_{period}'] = rz(rsi, 60)

    # --- Relative Price Position ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct_z'] = rz((close - bb_mid) / (2 * bb_std + eps), 60)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)
    feat['vwap_dev_z'] = rz((close - vwap_20) / (close + eps), 60)

    # --- Volume Dynamics ---
    feat['vol_ratio_20'] = volume / (volume.rolling(20).mean() + eps)
    feat['volume_z'] = rz(np.log(volume + eps), 60)

    # --- Market Character (Efficiency) ---
    for w in [10, 30]:
        direction = (close - close.shift(w)).abs()
        volatility = close.diff().abs().rolling(w).sum()
        feat[f'efficiency_{w}'] = direction / (volatility + eps)

    # --- Price Action / Wicks ---
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / (high - low + eps)
    feat['body_ratio'] = (close - open_).abs() / (high - low + eps)
    feat['hl_range_z'] = rz((high - low) / (close + eps), 60)

    # --- Interactions ---
    feat['mom_x_vol'] = feat['ret_z_3'] * feat['vol_ratio']
    feat['mom_x_eff'] = feat['ret_z_3'] * feat['efficiency_10']

    # --- Time & Context ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    if 'funding_rate' in df.columns:
        feat['funding_z'] = rz(df['funding_rate'], 60)
        feat['funding_mom'] = df['funding_rate'].diff(4)

    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=2):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + eps)
            except: return 0.0
        feat['obi'] = df.apply(lambda r: obi(r, 2), axis=1)

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
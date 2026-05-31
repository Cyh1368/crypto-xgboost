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
    "min_child_weight": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
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
    eps = 1e-9

    def rz(x, w):
        return (x - x.rolling(w).mean()) / (x.rolling(w).std() + eps)

    # --- Rolling Z-Scored Returns (Regime-Invariant) ---
    for lag in [1, 3, 5, 12]:
        r = close.pct_change(lag)
        feat[f'ret_z_{lag}'] = rz(r, 100)

    # --- Volatility Dynamics ---
    for w in [10, 30]:
        v = log_ret.rolling(w).std()
        feat[f'vol_z_{w}'] = rz(v, 100)

    vol_10 = log_ret.rolling(10).std()
    vol_60 = log_ret.rolling(60).std()
    feat['vol_ratio'] = vol_10 / (vol_60 + eps)
    feat['vol_regime'] = vol_10 / (vol_10.rolling(200).mean() + eps)

    # --- Kaufman Efficiency Ratio (Market Character) ---
    for w in [10, 30]:
        direction = (close - close.shift(w)).abs()
        volatility = close.diff().abs().rolling(w).sum()
        feat[f'efficiency_{w}'] = direction / (volatility + eps)

    # --- Bounded Technicals ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat['rsi_14'] = 100 - 100 / (1 + gain / (loss + eps))

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + eps)
    feat['vwap_dev_z'] = rz((close - (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)) / (close + eps), 100)

    # --- Volume Z-Score ---
    log_v = np.log(volume + 1)
    feat['volume_z'] = rz(log_v, 100)

    # --- Price Action / Wick Details ---
    feat['wick_upper'] = (high - df[['open', 'close']].max(axis=1)) / (high - low + eps)
    feat['wick_lower'] = (df[['open', 'close']].min(axis=1) - low) / (high - low + eps)
    feat['body_ratio'] = (close - df['open']).abs() / (high - low + eps)

    # --- Relative Range ---
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_z'] = rz(tr / close, 100)

    # --- Interactions (Normalized) ---
    feat['mom_vol_z'] = feat['ret_z_3'] * feat['vol_z_10']
    feat['mom_eff_z'] = feat['ret_z_3'] * feat['efficiency_10']

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- External Data (if available) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_z'] = rz(fr, 100)
        feat['funding_mom'] = fr.diff(4)

    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=2):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + eps)
            except: return 0.0
        feat['obi'] = df.apply(lambda r: obi(r), axis=1)

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
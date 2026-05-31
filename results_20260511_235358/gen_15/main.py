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
    "n_estimators": 1000,
    "max_depth": 4,
    "min_child_weight": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.20     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with focus on regime-invariant Z-scores and rate-of-change.
    """
    feat = pd.DataFrame(index=df.index)
    close, volume, high, low = df['close'], df['volume'], df['high'], df['low']

    # --- Returns & Local Z-Score Returns ---
    for lag in [1, 3, 5, 10, 20]:
        r = close.pct_change(lag)
        feat[f'ret_{lag}'] = r
        # Normalize returns by their own rolling volatility (regime-invariant)
        feat[f'ret_{lag}_z'] = (r - r.rolling(30).mean()) / (r.rolling(30).std() + 1e-9)

    # --- Volatility Ratios (Trend of Volatility) ---
    log_ret = np.log(close / close.shift(1))
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    vol_40 = log_ret.rolling(40).std()
    feat['vol_trend_short'] = vol_5 / (vol_20 + 1e-9)
    feat['vol_trend_long'] = vol_10 / (vol_40 + 1e-9)
    feat['vol_change'] = vol_5.diff(3) / (vol_20 + 1e-9)

    # --- Efficiency ratio (Kaufman) & Trend ---
    eff_15 = (close - close.shift(15)).abs() / (close.diff().abs().rolling(15).sum() + 1e-9)
    feat['efficiency_15'] = eff_15
    feat['efficiency_trend'] = eff_15.diff(5)

    # --- Relative Price Position (Z-Scored) ---
    for w in [15, 30]:
        mid = close.rolling(w).mean()
        std = close.rolling(w).std()
        feat[f'z_score_{w}'] = (close - mid) / (std + 1e-9)

    # --- RSI ---
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        feat[f'rsi_{period}'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # --- Volume Dynamics ---
    v_ma_5 = volume.rolling(5).mean()
    v_ma_20 = volume.rolling(20).mean()
    feat['volu_ratio'] = volume / (v_ma_20 + 1e-9)
    feat['volu_trend'] = v_ma_5 / (v_ma_20 + 1e-9)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev_z'] = (close - vwap) / (close.rolling(20).std() + 1e-9)

    # --- Interaction: Momentum / Volatility ---
    # In low vol, a 1% move is a signal. In high vol, it's noise.
    feat['mom_vol_adj'] = close.pct_change(5) / (vol_20 + 1e-9)

    # --- Candle Structure ---
    feat['range_norm'] = (high - low) / (close.rolling(20).mean() + 1e-9)
    feat['body_ratio'] = (close - df['open']).abs() / (high - low + 1e-9)
    feat['wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / (high - low + 1e-9)

    # --- Timing ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    # --- Orderbook & Funding (if available) ---
    if 'bids' in df.columns:
        def obi(row):
            b, a = np.array(row['bids']), np.array(row['asks'])
            return (np.sum(b[:3, 1]) - np.sum(a[:3, 1])) / (np.sum(b[:3, 1]) + np.sum(a[:3, 1]) + 1e-9)
        feat['obi_simple'] = df.apply(obi, axis=1)

    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_z'] = (fr - fr.rolling(40).mean()) / (fr.rolling(40).std() + 1e-9)
        feat['funding_mom'] = fr.diff(4)

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
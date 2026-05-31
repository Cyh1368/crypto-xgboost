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
    "n_estimators": 200,
    "max_depth": 3,
    "min_child_weight": 15,
    "reg_alpha": 0.1,
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

CLIP_PERCENTILE = 97
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-invariant features using rolling Z-scores and short lookbacks.
    """
    feat = pd.DataFrame(index=df.index)
    close, volume, high, low = df['close'], df['volume'], df['high'], df['low']
    log_ret = np.log(close / close.shift(1))

    def zscore(s, w=40):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)

    # --- Returns (Z-scored) ---
    for lag in [1, 3, 8]:
        feat[f'ret_{lag}_z'] = zscore(np.log(close / close.shift(lag)))

    # --- Volatility (Z-scored) ---
    for w in [5, 20]:
        feat[f'vol_{w}_z'] = zscore(log_ret.rolling(w).std())

    # Volatility Trend
    feat['vol_trend'] = log_ret.rolling(5).std() / (log_ret.rolling(20).std() + 1e-9)

    # --- RSI (Z-scored) ---
    for period in [10]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
        feat[f'rsi_{period}_z'] = zscore(rsi)

    # --- MACD (Z-scored) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_sig = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    feat['macd_sig_z'] = zscore(macd_sig)

    # --- Volume (Z-scored) ---
    feat['vol_ratio_z'] = zscore(volume / (volume.rolling(20).mean() + 1e-9))
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev_z'] = zscore((close - vwap) / (close + 1e-9))

    # --- Market Character (Efficiency & Autocorr) ---
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['autocorr_10'] = log_ret.rolling(10).apply(lambda x: x.autocorr() if x.std() > 0 else 0, raw=False)

    # --- Momentum / Vol Interaction ---
    feat['mom_regime'] = np.log(close / close.shift(5)) / (log_ret.rolling(5).std() + 1e-9)

    # --- Price Position ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct_z'] = zscore((close - bb_mid) / (2 * bb_std + 1e-9))

    low_20, high_20 = low.rolling(20).min(), high.rolling(20).max()
    feat['close_pos_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # --- Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook & Funding (Z-scored) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row):
            b, a = np.array(row['bids']), np.array(row['asks'])
            bv, av = np.sum(b[:, 1]), np.sum(a[:, 1])
            return (bv - av) / (bv + av + 1e-9)
        feat['obi_z'] = zscore(df.apply(obi, axis=1))

    if 'funding_rate' in df.columns:
        feat['funding_z'] = zscore(df['funding_rate'])
        feat['funding_mom'] = df['funding_rate'].diff(4)

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
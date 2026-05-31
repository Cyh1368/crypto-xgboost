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
    "min_child_weight": 25,
    "reg_alpha": 1.0,
    "reg_lambda": 15.0,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.03,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.22     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with rolling z-score normalization for regime invariance.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    eps = 1e-9

    def rz(x, w):
        return (x - x.rolling(w).mean()) / (x.rolling(w).std() + eps)

    # --- Returns & Momentum (Z-scored) ---
    log_ret = np.log(close / close.shift(1))
    for lag in [1, 3, 5, 10]:
        r = close.pct_change(lag)
        feat[f'ret_{lag}_z20'] = rz(r, 20)
        feat[f'ret_{lag}_z60'] = rz(r, 60)

    # --- Volatility Structure ---
    for w in [10, 20, 40]:
        vol = log_ret.rolling(w).std()
        feat[f'vol_{w}_z60'] = rz(vol, 60)

    vol_10 = log_ret.rolling(10).std()
    vol_60 = log_ret.rolling(60).std()
    feat['vol_trend'] = vol_10 / (vol_60 + eps)
    feat['vol_regime'] = vol_10 / (vol_10.rolling(200).mean() + eps)

    # --- Efficiency Ratio (Kaufman's ER) ---
    for w in [10, 30]:
        direction = (close - close.shift(w)).abs()
        volatility = close.diff().abs().rolling(w).sum()
        feat[f'eff_{w}'] = direction / (volatility + eps)

    # --- RSI (Z-scored) ---
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rsi = 100 - 100 / (1 + gain / (loss + eps))
        feat[f'rsi_{period}_z40'] = rz(rsi, 40)

    # --- Relative Price Position ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + eps)
    feat['price_sma20_dev_z40'] = rz((close - bb_mid) / (bb_mid + eps), 40)

    # --- Volume Dynamics ---
    feat['vol_ratio_20'] = volume / (volume.rolling(20).mean() + eps)
    feat['vol_z40'] = rz(volume, 40)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)
    feat['vwap_dev'] = (close - vwap_20) / (close + eps)

    # --- Candle Structure ---
    feat['hl_range_z20'] = rz((high - low) / (close + eps), 20)
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + eps)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + eps)

    # --- Interactions ---
    feat['mom_vol_interaction'] = feat['ret_3_z20'] * feat['vol_regime']
    feat['mom_eff_interaction'] = feat['ret_3_z20'] * feat['eff_10']

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    # --- Orderbook & Funding ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + eps)
            except: return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate_z40'] = rz(fr, 40)
        feat['funding_mom'] = fr.diff(4)

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
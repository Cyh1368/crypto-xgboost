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
    "n_estimators": 600,
    "max_depth": 3,
    "min_child_weight": 40,
    "reg_alpha": 1.5,
    "reg_lambda": 7.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.04,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-invariant features using rolling Z-scores and normalized indicators.
    """
    feat = pd.DataFrame(index=df.index)
    close, volume, high, low = df['close'], df['volume'], df['high'], df['low']
    log_ret = np.log(close / close.shift(1))

    def z_score(s, window=200):
        return (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-9)

    # --- Returns (Z-scored for regime invariance) ---
    for lag in [1, 3, 5, 12]:
        feat[f'ret_z_{lag}'] = z_score(close.pct_change(lag), 200)

    # --- Volatility & Regime ---
    vol_10 = log_ret.rolling(10).std()
    vol_50 = log_ret.rolling(50).std()
    feat['vol_z_10'] = z_score(vol_10, 200)
    feat['vol_trend'] = vol_10 / (vol_50 + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)

    # --- Momentum / Volatility Interaction ---
    feat['mom_vol_adj'] = feat['ret_z_3'] / (vol_10 + 1e-9)

    # --- Technical Indicators (Normalized) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    bb_mid = close.rolling(20).mean()
    feat['bb_pct'] = (close - bb_mid) / (2 * close.rolling(20).std() + 1e-9)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_norm'] = tr.rolling(14).mean() / (close + 1e-9)

    # --- Volume (Z-scored) ---
    feat['rel_vol_z'] = z_score(volume / (volume.rolling(20).mean() + 1e-9), 200)

    # --- Price Position ---
    feat['price_z_sma50'] = z_score((close - close.rolling(50).mean()) / (close.rolling(50).mean() + 1e-9), 200)
    low_20, high_20 = low.rolling(20).min(), high.rolling(20).max()
    feat['close_pos_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # --- Candle Structure ---
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)

    # --- Time Features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook & Funding (Conditional) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + 1e-9)
            except: return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)

    if 'funding_rate' in df.columns:
        feat['funding_rate'] = df['funding_rate']
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
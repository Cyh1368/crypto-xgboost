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
    "n_estimators": 1200,
    "max_depth": 7,
    "min_child_weight": 0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 99.0
MIN_PRED_STD_RATIO = 0.15
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features use only past data (shifted by 1).
    """
    feat = pd.DataFrame(index=df.index)
    eps = 1e-9

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    # --- Returns & Momentum ---
    log_close = np.log(close.clip(lower=eps))
    log_ret1 = log_close.diff()

    for w in [1, 3, 6, 12, 24, 48]:
        feat[f'ret_{w}'] = close.pct_change(w).shift(1)
        feat[f'lret_{w}'] = log_close.diff(w).shift(1)

    # --- Volatility & Z-Scores ---
    for w in [10, 30, 60]:
        feat[f'rv_{w}'] = log_ret1.rolling(w).std().shift(1)
        mu = log_ret1.rolling(w).mean().shift(1)
        sd = log_ret1.rolling(w).std().shift(1)
        feat[f'ret_z_{w}'] = (log_ret1.shift(1) - mu) / (sd + eps)

    # --- RSI ---
    for w in [14, 24]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean().shift(1)
        loss = (-delta.clip(upper=0)).rolling(w).mean().shift(1)
        feat[f'rsi_{w}'] = 100 - 100 / (1 + (gain / (loss + eps)))

    # --- Candle Anatomy ---
    candle_range = (high - low).clip(lower=eps)
    feat['body_to_range'] = ((close - close.shift(1)).abs() / candle_range).shift(1)
    feat['upper_wick_pct'] = ((high - np.maximum(close, close.shift(1))) / candle_range).shift(1)
    feat['lower_wick_pct'] = ((np.minimum(close, close.shift(1)) - low) / candle_range).shift(1)
    feat['close_pos_range'] = ((close - low) / candle_range).shift(1)

    # --- Volume & VWAP ---
    for w in [12, 24]:
        v_mu = volume.rolling(w).mean().shift(1)
        v_sd = volume.rolling(w).std().shift(1)
        feat[f'vol_z_{w}'] = (volume.shift(1) - v_mu) / (v_sd + eps)
        feat[f'vol_ratio_{w}'] = volume.shift(1) / (v_mu + eps)

    vwap20 = (close * volume).rolling(20).sum().shift(1) / (volume.rolling(20).sum().shift(1) + eps)
    feat['vwap_dev'] = (close.shift(1) - vwap20) / (close.shift(1) + eps)

    # --- Trend & Regime ---
    ma_fast = close.rolling(12).mean().shift(1)
    ma_slow = close.rolling(48).mean().shift(1)
    feat['trend_rel'] = (ma_fast / (ma_slow + eps)) - 1.0
    feat['vol_regime'] = feat['rv_10'] / (feat['rv_60'] + eps)

    # --- Short-term mean reversion & directional bias ---
    # Price percentile ranks for quick reversal detection
    feat['close_rank_8'] = close.rolling(8).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).shift(1)
    feat['close_rank_5'] = close.rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).shift(1)

    # Short-term momentum acceleration (directional bias)
    feat['ret_accel_3_12'] = (close.pct_change(3) - close.pct_change(12)).shift(1)
    feat['ret_accel_6_24'] = (close.pct_change(6) - close.pct_change(24)).shift(1)

    # Price velocity: 4-bar price change as leading signal
    feat['velocity_4'] = close.pct_change(4).shift(1)
    feat['velocity_8'] = close.pct_change(8).shift(1)

    # Return skewness for tail-risk detection (short window)
    ret_mu_10 = log_ret1.rolling(10).mean().shift(1)
    ret_sd_10 = log_ret1.rolling(10).std().shift(1)
    feat['ret_skew_10'] = (((log_ret1 - ret_mu_10) / (ret_sd_10 + eps)) ** 3).rolling(10).mean().shift(1)

    # --- Session / Time (No shift needed for time) ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook Imbalance ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def get_obi(row, tau=1.0):
            try:
                b = np.array(row['bids'], dtype=float)
                a = np.array(row['asks'], dtype=float)
                bv = np.sum(b[:, 1] * np.exp(-tau * np.arange(len(b))))
                av = np.sum(a[:, 1] * np.exp(-tau * np.arange(len(a))))
                return (bv - av) / (bv + av + eps)
            except: return 0.0
        feat['obi_t1'] = df.apply(lambda r: get_obi(r, 1.0), axis=1).shift(1)
        feat['obi_t3'] = df.apply(lambda r: get_obi(r, 3.0), axis=1).shift(1)

    # --- Funding Rate ---
    if 'funding_rate' in df.columns:
        fr = pd.to_numeric(df['funding_rate'], errors='coerce')
        feat['funding_rate'] = fr.shift(1)
        feat['funding_z'] = ((fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + eps)).shift(1)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    if feat.shape[1] > 80: feat = feat.iloc[:, :80]
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
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
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
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
    close, volume, high, low = df['close'], df['volume'], df['high'], df['low']
    log_ret = np.log(close / close.shift(1))

    # Rolling Z-score helper to ensure regime-invariance
    def z_score(s, w):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)

    # --- Returns & Momentum ---
    for lag in [1, 3, 6, 12, 24]:
        feat[f'ret_{lag}'] = close.pct_change(lag)
    feat['ret_z_20'] = z_score(close.pct_change(1), 20)
    feat['ret_z_40'] = z_score(close.pct_change(1), 40)

    # --- Volatility & Regime ---
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_10_z'] = z_score(vol_10, 60)
    feat['vol_20_z'] = z_score(vol_20, 60)
    feat['vol_roc'] = vol_10.diff(5) / (vol_10.shift(5) + 1e-9)
    feat['vol_regime'] = vol_10 / (vol_10.rolling(60).mean() + 1e-9)
    feat['vol_ret_corr_10'] = log_ret.rolling(10).corr(volume.pct_change(1))

    # --- RSI & Oscillators ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    feat['rsi_14_z'] = z_score(rsi, 60)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['atr_norm'] = log_ret.rolling(14).std() # Simplified proxy

    # --- Price & Volume Deviations ---
    sma10, sma20 = close.rolling(10).mean(), close.rolling(20).mean()
    feat['price_dev_10_z'] = z_score((close - sma10) / (sma10 + 1e-9), 60)
    feat['price_dev_20_z'] = z_score((close - sma20) / (sma20 + 1e-9), 60)

    v_ratio = volume / (volume.rolling(20).mean() + 1e-9)
    feat['vol_ratio_z'] = z_score(v_ratio, 60)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Interactions & Efficiency ---
    feat['mom_vol_interaction'] = close.pct_change(3) * feat['vol_regime']
    feat['mom_regime'] = close.pct_change(3) / (vol_10 + 1e-9)
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)

    # --- Range & Candle Structure ---
    hl_range = (high - low) / (close + 1e-9)
    feat['hl_range_z'] = z_score(hl_range, 60)
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['close_pos_10'] = (close - low.rolling(10).min()) / (high.rolling(10).max() - low.rolling(10).min() + 1e-9)
    feat['close_pos_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Funding & Orderbook ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_z'] = z_score(fr, 60)
        feat['funding_mom'] = fr.diff(4)
        feat['funding_x_vol'] = feat['funding_z'] * feat['vol_regime']

    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                bv = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                av = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bv - av) / (bv + av + 1e-9)
            except: return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

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
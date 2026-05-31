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
    "max_depth": 7,
    "min_child_weight": 2,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "gamma": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bynode": 0.8,
    "learning_rate": 0.04,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 60,
}

CLIP_PERCENTILE = 99.0
MIN_PRED_STD_RATIO = 0.15
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build strictly past-only feature matrix.
    Every feature at index t uses data from index t-1 or earlier.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    eps = 1e-9

    # Shift base data once to ensure no lookahead
    s_close = close.shift(1)
    s_high = high.shift(1)
    s_low = low.shift(1)
    s_vol = volume.shift(1)

    log_close = np.log(s_close.clip(lower=eps))
    log_ret1 = log_close.diff()

    # --- Returns & Momentum ---
    for lag in [1, 2, 3, 6, 12, 24]:
        feat[f"logret_{lag}"] = log_close.diff(lag)

    for w in [8, 16, 32]:
        feat[f"mom_{w}"] = s_close.pct_change(w)
        feat[f"ret_std_{w}"] = log_ret1.rolling(w).std()

    # --- Volatility & Compression ---
    tr = pd.concat([
        (s_high - s_low),
        (s_high - s_close.shift(1)).abs(),
        (s_low - s_close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    for w in [10, 20, 60]:
        feat[f"atr_{w}"] = tr.rolling(w).mean()
        feat[f"rv_{w}"] = log_ret1.rolling(w).std()

    feat["vol_regime"] = feat["rv_20"] / (feat["rv_60"].rolling(20).mean() + eps)
    feat["vol_compress"] = feat["rv_10"] / (feat["rv_60"] + eps)

    # --- Indicators ---
    for w in [7, 14]:
        delta = s_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / (loss + eps)
        feat[f"rsi_{w}"] = 100 - (100 / (1 + rs))

    for w in [24, 48, 96]:
        ma = s_close.rolling(w).mean()
        sd = s_close.rolling(w).std()
        feat[f"z_close_{w}"] = (s_close - ma) / (sd + eps)
        feat[f"dist_ma_{w}"] = (s_close / (ma + eps)) - 1.0

    ema12 = s_close.ewm(span=12, adjust=False).mean()
    ema26 = s_close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = macd - signal
    feat["macd_slope"] = macd.diff()

    # --- Candle Anatomy ---
    s_range = (s_high - s_low).clip(lower=eps)
    feat["body_pct"] = (s_close - s_close.shift(1)).abs() / (s_range + eps)
    feat["upper_wick"] = (s_high - s_close.clip(lower=s_close.shift(1))) / (s_range + eps)
    feat["lower_wick"] = (s_close.clip(upper=s_close.shift(1)) - s_low) / (s_range + eps)
    feat["close_pos"] = (s_close - s_low) / (s_range + eps)
    feat["efficiency_10"] = (s_close - s_close.shift(10)).abs() / (tr.rolling(10).sum() + eps)

    # --- Volume ---
    vma24 = s_vol.rolling(24).mean()
    feat["vol_z_24"] = (s_vol - vma24) / (s_vol.rolling(24).std() + eps)
    feat["vol_surge"] = s_vol / (vma24 + eps)
    vwap20 = (s_close * s_vol).rolling(20).sum() / (s_vol.rolling(20).sum() + eps)
    feat["vwap_dev"] = (s_close - vwap20) / (s_close + eps)

    # --- Interactions & Higher-level ---
    feat["trend_x_vol"] = (ema12/ema26 - 1.0) * feat["vol_regime"]
    feat["breakout_strength"] = feat["close_pos"] * feat["vol_surge"]
    feat["hl_rank_48"] = s_close.rolling(48).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    feat["trend_accel"] = s_close.pct_change(24) - s_close.pct_change(48)

    # --- Session ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Microstructure (shifted) ---
    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1.0):
            try:
                b, a = np.array(row["bids"], dtype=float), np.array(row["asks"], dtype=float)
                bv = np.sum(b[:, 1] * np.exp(-tau * np.arange(len(b))))
                av = np.sum(a[:, 1] * np.exp(-tau * np.arange(len(a))))
                return (bv - av) / (bv + av + eps)
            except: return 0.0
        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1.0), axis=1).shift(1)
        feat["obi_diff"] = feat["obi_tau1"] - df.apply(lambda r: obi(r, 3.0), axis=1).shift(1)

    if "funding_rate" in df.columns:
        fr = pd.to_numeric(df["funding_rate"], errors="coerce")
        feat["funding_val"] = fr.shift(1)
        feat["funding_z"] = (fr.shift(1) - fr.rolling(32).mean().shift(1)) / (fr.rolling(32).std().shift(1) + eps)
        feat["funding_delta"] = fr.diff().shift(1)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat.iloc[:, :78] # Safe margin under 80
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
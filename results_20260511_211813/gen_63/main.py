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
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.25,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.85,
    "learning_rate": 0.04,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 80,
}

CLIP_PERCENTILE = 98.0
MIN_PRED_STD_RATIO = 0.15
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features use only past data (explicitly shifted where needed).
    Returns a DataFrame aligned to df.index.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    eps = 1e-9

    # Base series shifted once for strict past-only usage
    s_close = close.shift(1)
    s_high = high.shift(1)
    s_low = low.shift(1)
    s_vol = volume.shift(1)

    log_close = np.log(s_close.clip(lower=eps))
    log_ret1 = log_close.diff()

    # --- Short/medium horizon returns and momentum ---
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        feat[f"ret_{lag}"] = s_close.pct_change(lag)
        feat[f"logret_{lag}"] = log_close.diff(lag)

    for w in [4, 8, 16, 32]:
        feat[f"mom_{w}"] = s_close.pct_change(w)
        feat[f"mom_slope_{w}"] = s_close.pct_change(w).diff()
        feat[f"ret_mean_{w}"] = log_ret1.rolling(w).mean()
        feat[f"ret_std_{w}"] = log_ret1.rolling(w).std()

    # --- Volatility / regime ---
    tr = pd.concat(
        [
            (s_high - s_low),
            (s_high - s_close.shift(1)).abs(),
            (s_low - s_close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for w in [5, 10, 20, 60]:
        feat[f"atr_{w}"] = tr.rolling(w).mean()
        feat[f"rv_{w}"] = log_ret1.rolling(w).std()

    feat["atr_norm_20"] = feat["atr_20"] / (s_close + eps)
    feat["vol_regime"] = feat["rv_20"] / (feat["rv_60"].rolling(20).mean() + eps)
    feat["vol_compress"] = feat["rv_10"] / (feat["rv_60"] + eps)
    feat["vol_change_20"] = feat["rv_20"] / (feat["rv_20"].rolling(20).mean() + eps)

    # --- Trend / mean reversion position ---
    for w in [12, 24, 48, 96]:
        ma = s_close.rolling(w).mean()
        sd = s_close.rolling(w).std()
        feat[f"z_close_{w}"] = (s_close - ma) / (sd + eps)
        feat[f"dist_ma_{w}"] = (s_close / (ma + eps)) - 1.0

    feat["ema_fast"] = s_close.ewm(span=8, adjust=False).mean()
    feat["ema_slow"] = s_close.ewm(span=21, adjust=False).mean()
    feat["ema_trend"] = (feat["ema_fast"] / (feat["ema_slow"] + eps)) - 1.0
    feat["ema_spread"] = (feat["ema_fast"] - feat["ema_slow"]) / (s_close + eps)

    ema12 = s_close.ewm(span=12, adjust=False).mean()
    ema26 = s_close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = macd - signal
    feat["macd_slope"] = macd.diff()

    # --- Candle anatomy / range efficiency ---
    candle_range = (s_high - s_low).clip(lower=eps)
    body = (s_close - s_close.shift(1)).abs()
    upper_wick = (s_high - np.maximum(s_close, s_close.shift(1))).clip(lower=0)
    lower_wick = (np.minimum(s_close, s_close.shift(1)) - s_low).clip(lower=0)

    feat["range_pct"] = candle_range / (s_close + eps)
    feat["body_to_range"] = body / (candle_range + eps)
    feat["upper_wick_pct"] = upper_wick / (candle_range + eps)
    feat["lower_wick_pct"] = lower_wick / (candle_range + eps)
    feat["close_pos_range"] = (s_close - s_low) / (candle_range + eps)
    feat["efficiency_8"] = (s_close - s_close.shift(8)).abs() / (tr.rolling(8).sum() + eps)

    # --- Volume / participation ---
    for w in [12, 24, 48]:
        vma = s_vol.rolling(w).mean()
        vsd = s_vol.rolling(w).std()
        feat[f"vol_z_{w}"] = (s_vol - vma) / (vsd + eps)
        feat[f"vol_ratio_{w}"] = s_vol / (vma + eps)

    feat["vol_surge"] = feat["vol_ratio_24"] / (feat["vol_ratio_48"].rolling(12).mean() + eps)
    vwap20 = (s_close * s_vol).rolling(20).sum() / (s_vol.rolling(20).sum() + eps)
    feat["vwap_dev_20"] = (s_close - vwap20) / (s_close + eps)

    # --- Robust normalized returns ---
    for w in [16, 32, 64]:
        mu = log_ret1.rolling(w).mean()
        sd = log_ret1.rolling(w).std()
        feat[f"ret_z_{w}"] = (log_ret1 - mu) / (sd + eps)

    # --- Compact regime interactions ---
    feat["trend_x_vol"] = feat["ema_trend"] * feat["vol_regime"]
    feat["trend_x_volume"] = feat["ema_trend"] * feat["vol_z_24"]
    feat["meanrev_x_vol"] = feat["z_close_24"] * feat["vol_compress"]
    feat["momentum_x_range"] = feat["mom_16"] * feat["range_pct"]
    feat["momentum_x_wick"] = feat["mom_16"] * (feat["upper_wick_pct"] - feat["lower_wick_pct"])
    feat["breakout_pressure"] = feat["close_pos_range"] * feat["vol_surge"] * (feat["range_pct"] / (feat["atr_norm_20"] + eps))
    feat["volatility_breakout"] = feat["vol_change_20"] * feat["breakout_pressure"]

    # --- Session / time ---
    hour = df.index.hour
    dow = df.index.dayofweek
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feat["is_weekend"] = (dow >= 5).astype(float)

    # --- Higher-level regime feature ---
    trend_48 = s_close.pct_change(48)
    trend_96 = s_close.pct_change(96)
    feat["trend_accel"] = trend_48 - trend_96
    feat["trend_strength"] = trend_48.abs()
    feat["trend_to_vol"] = feat["trend_strength"] / (feat["rv_20"] + eps)

    # --- Orderbook imbalance (if available) ---
    if "bids" in df.columns and "asks" in df.columns:
        def book_imbalance(row, tau=1.0):
            try:
                bids = np.array(row["bids"], dtype=float)
                asks = np.array(row["asks"], dtype=float)
                if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                    return 0.0
                bvol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                avol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bvol - avol) / (bvol + avol + eps)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: book_imbalance(r, 1.0), axis=1).shift(1)
        feat["obi_tau3"] = df.apply(lambda r: book_imbalance(r, 3.0), axis=1).shift(1)
        feat["obi_diff"] = feat["obi_tau1"] - feat["obi_tau3"]

    # --- Funding rate ---
    if "funding_rate" in df.columns:
        fr = pd.to_numeric(df["funding_rate"], errors="coerce")
        feat["funding_rate"] = fr.shift(1)
        feat["funding_ma_8h"] = fr.rolling(32).mean().shift(1)
        feat["funding_z_8h"] = (fr.shift(1) - fr.rolling(32).mean().shift(1)) / (fr.rolling(32).std().shift(1) + eps)
        feat["funding_change"] = fr.diff().shift(1)
        feat["funding_x_trend"] = feat["funding_rate"] * feat["trend_accel"]
        feat["funding_x_obi"] = feat["funding_rate"] * feat.get("obi_diff", 0.0)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Keep a compact, information-dense subset under 80 features
    keep_cols = [
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
        "mom_4", "mom_8", "mom_16", "mom_32",
        "mom_slope_4", "mom_slope_8", "mom_slope_16",
        "ret_mean_16", "ret_std_16", "ret_mean_32", "ret_std_32",
        "atr_5", "atr_10", "atr_20", "rv_10", "rv_20", "rv_60",
        "atr_norm_20", "vol_regime", "vol_compress", "vol_change_20",
        "z_close_12", "z_close_24", "z_close_48", "dist_ma_12", "dist_ma_24",
        "ema_trend", "ema_spread", "macd_hist", "macd_slope",
        "range_pct", "body_to_range", "upper_wick_pct", "lower_wick_pct",
        "close_pos_range", "efficiency_8",
        "vol_z_12", "vol_z_24", "vol_z_48", "vol_ratio_12", "vol_ratio_24", "vol_ratio_48",
        "vol_surge", "vwap_dev_20",
        "ret_z_16", "ret_z_32", "ret_z_64",
        "trend_x_vol", "trend_x_volume", "meanrev_x_vol",
        "momentum_x_range", "momentum_x_wick", "breakout_pressure", "volatility_breakout",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
        "trend_accel", "trend_strength", "trend_to_vol",
        "obi_tau1", "obi_tau3", "obi_diff",
        "funding_rate", "funding_ma_8h", "funding_z_8h", "funding_change",
        "funding_x_trend", "funding_x_obi",
    ]
    keep_cols = [c for c in keep_cols if c in feat.columns]
    feat = feat[keep_cols]

    if feat.shape[1] > 80:
        feat = feat.iloc[:, :80]

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
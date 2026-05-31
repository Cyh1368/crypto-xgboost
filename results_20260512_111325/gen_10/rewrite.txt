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
    "n_estimators": 260,
    "max_depth": 3,
    "min_child_weight": 8,
    "reg_alpha": 0.8,
    "reg_lambda": 4.0,
    "subsample": 0.75,
    "colsample_bytree": 0.72,
    "learning_rate": 0.05,
    "gamma": 0.15,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 35,
}

CLIP_PERCENTILE = 96
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact regime-adaptive feature set using only past data.
    Focus:
      - short lookbacks
      - rolling z-score normalization
      - trend-of-regime features
      - a few interaction terms
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    log_close = np.log(close.replace(0, np.nan))
    log_ret = log_close.diff()

    eps = 1e-9

    def rolling_z(s, w):
        m = s.rolling(w, min_periods=w).mean()
        sd = s.rolling(w, min_periods=w).std()
        return (s - m) / (sd + eps)

    def rolling_rank01(s, w):
        return s.rolling(w, min_periods=w).rank(pct=True)

    # --- Short-horizon returns and normalized momentum ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        r = close.pct_change(lag)
        feat[f"ret_{lag}"] = r
        if lag <= 8:
            feat[f"ret_z_{lag}_20"] = rolling_z(r, 20)
            feat[f"ret_z_{lag}_40"] = rolling_z(r, 40)

    # --- Rolling momentum strength, normalized to local regime ---
    mom_5 = close.pct_change(5)
    mom_10 = close.pct_change(10)
    feat["mom_5_z20"] = rolling_z(mom_5, 20)
    feat["mom_10_z40"] = rolling_z(mom_10, 40)

    # --- Realized volatility and its trend ---
    rv_5 = log_ret.rolling(5, min_periods=5).std()
    rv_10 = log_ret.rolling(10, min_periods=10).std()
    rv_20 = log_ret.rolling(20, min_periods=20).std()

    feat["rv_5"] = rv_5
    feat["rv_10"] = rv_10
    feat["rv_20"] = rv_20
    feat["rv_20_z60"] = rolling_z(rv_20, 60)
    feat["rv_trend_10"] = (rv_10 / (rv_10.shift(5) + eps)) - 1.0
    feat["rv_trend_20"] = (rv_20 / (rv_20.shift(10) + eps)) - 1.0
    feat["rv_rank_120"] = rolling_rank01(rv_20, 120)

    # --- Efficiency / chop regime ---
    net_10 = (close - close.shift(10)).abs()
    path_10 = close.diff().abs().rolling(10, min_periods=10).sum()
    net_20 = (close - close.shift(20)).abs()
    path_20 = close.diff().abs().rolling(20, min_periods=20).sum()

    feat["eff_10"] = net_10 / (path_10 + eps)
    feat["eff_20"] = net_20 / (path_20 + eps)
    feat["eff_trend"] = (feat["eff_10"] / (feat["eff_10"].shift(5) + eps)) - 1.0
    feat["eff_z_60"] = rolling_z(feat["eff_20"], 60)

    # --- Autocorrelation proxy: consecutive return persistence ---
    ret_1 = close.pct_change(1)
    feat["ret_persist_5"] = ret_1.rolling(5, min_periods=5).mean() / (ret_1.rolling(5, min_periods=5).std() + eps)
    feat["ret_persist_10"] = ret_1.rolling(10, min_periods=10).mean() / (ret_1.rolling(10, min_periods=10).std() + eps)
    feat["ret_persist_trend"] = feat["ret_persist_5"] - feat["ret_persist_10"]

    # --- Candle shape / position within range ---
    rng = (high - low).replace(0, np.nan)
    feat["body_ratio"] = (close - open_).abs() / (rng + eps)
    feat["upper_wick"] = (high - np.maximum(open_, close)) / (rng + eps)
    feat["lower_wick"] = (np.minimum(open_, close) - low) / (rng + eps)
    feat["close_pos"] = (close - low) / (rng + eps)

    feat["body_ratio_z40"] = rolling_z(feat["body_ratio"], 40)
    feat["close_pos_z40"] = rolling_z(feat["close_pos"], 40)

    # --- Short-term range expansion / contraction ---
    feat["hl_range"] = (high - low) / (close + eps)
    feat["hl_range_z20"] = rolling_z(feat["hl_range"], 20)
    feat["hl_range_trend"] = feat["hl_range"] / (feat["hl_range"].shift(5) + eps) - 1.0

    # --- Volume regime and volume shock ---
    vol_ma_5 = volume.rolling(5, min_periods=5).mean()
    vol_ma_20 = volume.rolling(20, min_periods=20).mean()
    vol_sd_20 = volume.rolling(20, min_periods=20).std()

    feat["vol_ratio_5"] = volume / (vol_ma_5 + eps)
    feat["vol_ratio_20"] = volume / (vol_ma_20 + eps)
    feat["vol_z_20"] = (volume - vol_ma_20) / (vol_sd_20 + eps)
    feat["vol_ratio_z60"] = rolling_z(feat["vol_ratio_20"], 60)
    feat["vol_trend"] = feat["vol_ratio_20"] / (feat["vol_ratio_20"].shift(5) + eps) - 1.0

    # --- Local price position / deviation from short moving averages ---
    sma_5 = close.rolling(5, min_periods=5).mean()
    sma_10 = close.rolling(10, min_periods=10).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()

    feat["sma5_dev"] = (close - sma_5) / (sma_5 + eps)
    feat["sma10_dev"] = (close - sma_10) / (sma_10 + eps)
    feat["sma20_dev"] = (close - sma_20) / (sma_20 + eps)
    feat["sma_stack"] = (sma_5 - sma_20) / (sma_20 + eps)

    feat["sma5_dev_z40"] = rolling_z(feat["sma5_dev"], 40)
    feat["sma20_dev_z60"] = rolling_z(feat["sma20_dev"], 60)

    # --- Short-volatility-adjusted momentum ---
    feat["mom_vol_5"] = mom_5 / (rv_20 + eps)
    feat["mom_vol_10"] = mom_10 / (rv_20 + eps)
    feat["mom_x_volreg"] = mom_5 * feat["rv_20_z60"]
    feat["mom_x_eff"] = mom_5 * feat["eff_20"]
    feat["mom_x_voltrend"] = mom_5 * feat["rv_trend_20"]

    # --- Regime change indicators ---
    feat["regime_shift_vol"] = feat["rv_20_z60"] - feat["rv_20_z60"].shift(5)
    feat["regime_shift_eff"] = feat["eff_z_60"] - feat["eff_z_60"].shift(5)
    feat["regime_shift_price"] = feat["sma20_dev_z60"] - feat["sma20_dev_z60"].shift(5)

    # --- Rolling standardized return shape ---
    feat["ret1_z20"] = rolling_z(ret_1, 20)
    feat["ret1_z40"] = rolling_z(ret_1, 40)
    feat["ret1_rank60"] = rolling_rank01(ret_1, 60)

    # --- Session / periodicity ---
    idx = df.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

    # --- Optional orderbook features ---
    if "bids" in df.columns and "asks" in df.columns:
        def ob_imbalance(row):
            try:
                bids = np.asarray(row["bids"], dtype=float)
                asks = np.asarray(row["asks"], dtype=float)
                if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                    return np.nan
                # emphasize top-of-book, decay with depth
                depth = min(len(bids), len(asks), 10)
                weights = np.exp(-0.35 * np.arange(depth))
                bid_vol = np.sum(bids[:depth, 1] * weights)
                ask_vol = np.sum(asks[:depth, 1] * weights)
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
            except Exception:
                return np.nan

        feat["obi_top10"] = df.apply(ob_imbalance, axis=1)
        feat["obi_z60"] = rolling_z(feat["obi_top10"], 60)
        feat["obi_x_mom"] = feat["obi_top10"] * mom_5

    # --- Optional funding features ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding"] = fr
        feat["funding_z64"] = rolling_z(fr, 64)
        feat["funding_trend"] = fr - fr.shift(4)
        feat["funding_x_vol"] = fr * feat["rv_20_z60"]
        feat["funding_x_mom"] = fr * mom_5

    # --- Final cleanup ---
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Keep feature count safely below 80
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
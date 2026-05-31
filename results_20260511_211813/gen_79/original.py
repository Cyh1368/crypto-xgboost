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
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 100,
}

CLIP_PERCENTILE = 95
MIN_PRED_STD_RATIO = 0.15
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

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    log_close = np.log(close.replace(0, np.nan))
    log_ret = log_close.diff()

    # ── Multi-scale returns and phase structure
    for lag in [1, 2, 4, 8, 16, 32]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    r1 = close.pct_change(1)
    r3 = close.pct_change(3)
    r6 = close.pct_change(6)
    r12 = close.pct_change(12)
    r24 = close.pct_change(24)

    feat["ret_accel_3"] = r1 - r3 / 3.0
    feat["ret_accel_6"] = r3 - r6 / 2.0
    feat["ret_accel_12"] = r6 - r12 / 2.0
    feat["trend_stack"] = np.sign(r1) + np.sign(r3) + np.sign(r6) + np.sign(r12)

    # ── EW trend / dispersion
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_mid = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=55, adjust=False).mean()

    feat["ema_gap_fast"] = (ema_fast - ema_mid) / (close + 1e-9)
    feat["ema_gap_slow"] = (ema_mid - ema_slow) / (close + 1e-9)
    feat["ema_slope_fast"] = ema_fast.pct_change(3)
    feat["ema_slope_slow"] = ema_slow.pct_change(8)
    feat["trend_ratio"] = (ema_fast - ema_slow) / (close.rolling(20).std() + 1e-9)

    # ── Volatility regime, compression, and expansion
    rv_8 = log_ret.rolling(8).std()
    rv_24 = log_ret.rolling(24).std()
    rv_96 = log_ret.rolling(96).std()
    rv_192 = log_ret.rolling(192).std()

    feat["rv_8"] = rv_8
    feat["rv_24"] = rv_24
    feat["rv_96"] = rv_96
    feat["vol_ratio_8_96"] = rv_8 / (rv_96 + 1e-9)
    feat["vol_ratio_24_192"] = rv_24 / (rv_192 + 1e-9)
    feat["vol_accel"] = rv_8 / (rv_24 + 1e-9) - 1.0
    feat["vol_ratio_2_8"] = log_ret.rolling(2).std() / (rv_8 + 1e-9)

    # A smooth compression/expansion state
    feat["vol_state"] = np.tanh((feat["vol_ratio_8_96"] - 1.0) * 1.5)

    # ── Candle geometry / microstructure of bar
    bar_range = (high - low).replace(0, np.nan)
    body = close - open_
    feat["candle_body"] = body / (bar_range + 1e-9)
    feat["upper_wick"] = (high - np.maximum(close, open_)) / (bar_range + 1e-9)
    feat["lower_wick"] = (np.minimum(close, open_) - low) / (bar_range + 1e-9)
    feat["range_norm"] = bar_range / (close + 1e-9)
    feat["close_loc"] = (close - low) / (bar_range + 1e-9)

    # ── Range position / breakout pressure
    hi_20 = high.rolling(20).max()
    lo_20 = low.rolling(20).min()
    hi_60 = high.rolling(60).max()
    lo_60 = low.rolling(60).min()
    feat["range_pos_20"] = (close - lo_20) / ((hi_20 - lo_20) + 1e-9)
    feat["range_pos_60"] = (close - lo_60) / ((hi_60 - lo_60) + 1e-9)
    feat["range_width_20"] = (hi_20 - lo_20) / (close + 1e-9)
    feat["range_width_60"] = (hi_60 - lo_60) / (close + 1e-9)
    feat["bb_width_20"] = (hi_20 - lo_20) / (ema_mid + 1e-9)
    feat["price_vol_corr_20"] = r1.rolling(20).corr(volume.pct_change(1))
    feat["skew_20"] = r1.rolling(20).skew()

    # ── Mean reversion pressure
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    vwap_60 = (close * volume).rolling(60).sum() / (volume.rolling(60).sum() + 1e-9)
    feat["vwap_dev_20"] = (close - vwap_20) / (close.rolling(20).std() + 1e-9)
    feat["vwap_dev_60"] = (close - vwap_60) / (close.rolling(60).std() + 1e-9)

    # ── RSI / oscillatory state
    def rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    feat["rsi_7"] = rsi(close, 7)
    feat["rsi_14"] = rsi(close, 14)
    feat["rsi_28"] = rsi(close, 28)
    feat["rsi_slope"] = feat["rsi_7"].diff()
    feat["rsi_z_30"] = (feat["rsi_14"] - feat["rsi_14"].rolling(30).mean()) / (feat["rsi_14"].rolling(30).std() + 1e-9)

    # ── Volume state
    vol_mean_20 = volume.rolling(20).mean()
    vol_mean_60 = volume.rolling(60).mean()
    vol_std_20 = volume.rolling(20).std()

    feat["volume_ratio_5"] = volume / (volume.rolling(5).mean() + 1e-9)
    feat["volume_ratio_20"] = volume / (vol_mean_20 + 1e-9)
    feat["volume_ratio_60"] = volume / (vol_mean_60 + 1e-9)
    feat["volume_z_20"] = (volume - vol_mean_20) / (vol_std_20 + 1e-9)
    feat["volume_chg_1"] = volume.pct_change(1)
    feat["volume_chg_4"] = volume.pct_change(4)

    # ── Rolling quantile position: captures state without assuming linearity
    feat["ret_qpos_60"] = (r1.rolling(60).rank(pct=True))
    feat["vol_qpos_60"] = (volume.rolling(60).rank(pct=True))
    feat["close_qpos_60"] = (close.rolling(60).rank(pct=True))

    # ── Direction persistence / churn
    sign_r1 = np.sign(r1)
    feat["dir_persist_8"] = sign_r1.rolling(8).mean()
    feat["dir_persist_24"] = sign_r1.rolling(24).mean()
    feat["sign_flip_8"] = (sign_r1 != sign_r1.shift(1)).astype(float).rolling(8).mean()
    feat["sign_flip_24"] = (sign_r1 != sign_r1.shift(1)).astype(float).rolling(24).mean()

    # ── Time features
    idx = df.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    # ── Cross-regime interactions: only a few, but structurally different
    feat["trend_x_vol"] = feat["trend_ratio"] * feat["vol_state"]
    feat["momentum_x_vol"] = (feat["ret_1"] if "ret_1" in feat.columns else r1) * feat["vol_state"]
    feat["range_x_volume"] = feat["range_norm"] * feat["volume_ratio_20"]
    feat["rsi_x_trend"] = (feat["rsi_14"] / 100.0) * feat["trend_ratio"]
    feat["wick_imbalance"] = feat["lower_wick"] - feat["upper_wick"]
    feat["breakout_pressure"] = feat["range_pos_20"] * (feat["volume_ratio_5"] - 1.0)

    # ── Orderbook features, if present
    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row["bids"], dtype=float)
                asks = np.array(row["asks"], dtype=float)
                if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                    return 0.0
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau3"] = df.apply(lambda r: obi(r, 3), axis=1)

    # ── Funding features, if present
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_ma_8h"] = fr.rolling(32).mean()
        feat["funding_chg_1"] = fr.diff(1)
        feat["funding_chg_4"] = fr.diff(4)
        feat["funding_z_32"] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)

    # Clean
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Keep feature count comfortably below cap
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
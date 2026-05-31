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
    "n_estimators": 240,
    "max_depth": 4,
    "min_child_weight": 8,
    "reg_alpha": 0.25,
    "reg_lambda": 2.5,
    "gamma": 0.15,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 35,
}

CLIP_PERCENTILE = 97
MIN_PRED_STD_RATIO = 0.18
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

    eps = 1e-9
    log_close = np.log(close.replace(0, np.nan))
    log_ret = log_close.diff()
    ret1 = close.pct_change(1)

    def rmean(s, w):
        return s.rolling(w, min_periods=w).mean()

    def rstd(s, w):
        return s.rolling(w, min_periods=w).std()

    def rz(s, w):
        m = rmean(s, w)
        sd = rstd(s, w)
        return (s - m) / (sd + eps)

    # --- Short-horizon returns and momentum ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    feat["ret_1_z_20"] = rz(ret1, 20)
    feat["ret_1_z_60"] = rz(ret1, 60)
    feat["mom_3_10"] = close.pct_change(3) - close.pct_change(10)
    feat["mom_5_20"] = close.pct_change(5) - close.pct_change(20)

    # --- Volatility with regime normalization ---
    vol_5 = log_ret.rolling(5, min_periods=5).std()
    vol_10 = log_ret.rolling(10, min_periods=10).std()
    vol_20 = log_ret.rolling(20, min_periods=20).std()
    vol_40 = log_ret.rolling(40, min_periods=40).std()

    feat["vol_5"] = vol_5
    feat["vol_10"] = vol_10
    feat["vol_20"] = vol_20
    feat["vol_20_z_60"] = rz(vol_20, 60)
    feat["vol_10_z_60"] = rz(vol_10, 60)
    feat["vol_trend_20_40"] = (vol_20 / (vol_40 + eps)) - 1.0
    feat["vol_trend_5_20"] = (vol_5 / (vol_20 + eps)) - 1.0

    # --- Efficient price / trend quality ---
    abs_ret = ret1.abs()
    feat["efficiency_10"] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10, min_periods=10).sum() + eps)
    feat["efficiency_20"] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20, min_periods=20).sum() + eps)
    feat["efficiency_trend"] = feat["efficiency_10"] - feat["efficiency_20"]

    # --- Autocorrelation-like measures without lookahead ---
    feat["ret_autocorr_10"] = ret1.rolling(10, min_periods=10).corr(ret1.shift(1))
    feat["ret_autocorr_20"] = ret1.rolling(20, min_periods=20).corr(ret1.shift(1))
    feat["ret_autocorr_trend"] = feat["ret_autocorr_10"] - feat["ret_autocorr_20"]

    # --- RSI / normalized oscillators ---
    for period in [5, 8, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / (loss + eps)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    feat["rsi_5_z_20"] = rz(feat["rsi_5"], 20)
    feat["rsi_14_z_60"] = rz(feat["rsi_14"], 60)

    # --- Price position / mean reversion ---
    sma_5 = close.rolling(5, min_periods=5).mean()
    sma_10 = close.rolling(10, min_periods=10).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()

    feat["price_sma5_dev"] = (close - sma_5) / (sma_5 + eps)
    feat["price_sma10_dev"] = (close - sma_10) / (sma_10 + eps)
    feat["price_sma20_dev"] = (close - sma_20) / (sma_20 + eps)
    feat["sma5_sma20"] = (sma_5 - sma_20) / (sma_20 + eps)
    feat["bb_pct_20"] = (close - sma_20) / (2.0 * std_20 + eps)
    feat["close_position_20"] = (close - low.rolling(20, min_periods=20).min()) / (
        high.rolling(20, min_periods=20).max() - low.rolling(20, min_periods=20).min() + eps
    )

    # --- Candle / range features ---
    hl_range = (high - low) / (close + eps)
    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    feat["hl_range"] = hl_range
    feat["hl_range_z_20"] = rz(hl_range, 20)
    feat["candle_body_ratio"] = body / ((high - low) + eps)
    feat["upper_wick_ratio"] = upper_wick / ((high - low) + eps)
    feat["lower_wick_ratio"] = lower_wick / ((high - low) + eps)
    feat["range_body_pressure"] = (body / ((high - low) + eps)) * np.sign(close - open_)

    # --- Volume / participation ---
    vol_ma_5 = volume.rolling(5, min_periods=5).mean()
    vol_ma_20 = volume.rolling(20, min_periods=20).mean()
    vol_ma_60 = volume.rolling(60, min_periods=60).mean()
    feat["volume_ratio_5"] = volume / (vol_ma_5 + eps)
    feat["volume_ratio_20"] = volume / (vol_ma_20 + eps)
    feat["volume_ratio_60"] = volume / (vol_ma_60 + eps)
    feat["volume_z_20"] = rz(volume, 20)
    feat["volume_trend_20_60"] = (vol_ma_20 / (vol_ma_60 + eps)) - 1.0
    feat["vol_ret_corr_20"] = log_ret.rolling(20, min_periods=20).corr(volume.pct_change(1))

    # --- VWAP / pressure ---
    vwap_20 = (close * volume).rolling(20, min_periods=20).sum() / (volume.rolling(20, min_periods=20).sum() + eps)
    feat["vwap_dev"] = (close - vwap_20) / (close + eps)
    feat["vwap_dev_z_60"] = rz(feat["vwap_dev"], 60)

    # --- Market character trends ---
    feat["trend_5"] = close.pct_change(5)
    feat["trend_10"] = close.pct_change(10)
    feat["trend_20"] = close.pct_change(20)
    feat["trend_strength_10"] = feat["trend_10"].abs()
    feat["trend_strength_20"] = feat["trend_20"].abs()
    feat["trend_dir_10"] = np.sign(feat["trend_10"])
    feat["trend_dir_20"] = np.sign(feat["trend_20"])
    feat["trend_consistency"] = np.sign(ret1).rolling(10, min_periods=10).mean()

    # --- Regime features: robust, relative, and trend-based ---
    feat["vol_regime_20_60"] = vol_20 / (vol_40 + eps)
    feat["vol_regime_rank_60"] = vol_20.rolling(60, min_periods=60).rank(pct=True)
    feat["vol_regime_change"] = feat["vol_regime_20_60"] - feat["vol_regime_20_60"].shift(5)
    feat["range_regime_20"] = hl_range.rolling(20, min_periods=20).mean() / (
        hl_range.rolling(60, min_periods=60).mean() + eps
    )
    feat["efficiency_rank_60"] = feat["efficiency_20"].rolling(60, min_periods=60).rank(pct=True)
    feat["autocorr_rank_60"] = feat["ret_autocorr_20"].rolling(60, min_periods=60).rank(pct=True)

    # --- Feature interactions that often generalize better ---
    feat["momentum_vol_interaction"] = feat["trend_5"] * feat["vol_regime_20_60"]
    feat["momentum_regime_20"] = feat["trend_5"] * feat["vol_regime_rank_60"]
    feat["momentum_regime_fast"] = feat["trend_5"] * feat["vol_trend_5_20"]
    feat["price_vol_interaction"] = feat["price_sma10_dev"] * feat["vol_regime_20_60"]
    feat["volume_price_pressure"] = feat["volume_ratio_20"] * feat["price_sma10_dev"]
    feat["volume_bb_interaction"] = feat["volume_ratio_20"] * feat["bb_pct_20"]
    feat["rsi_vol_interaction"] = feat["rsi_5_z_20"] * feat["vol_regime_20_60"]

    # --- Time/session ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Funding rate (if present) ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_rate_z_60"] = rz(fr, 60)
        feat["funding_8h_ma"] = fr.rolling(32, min_periods=32).mean()
        feat["funding_momentum"] = fr.diff(4)
        feat["funding_x_volregime"] = fr * feat["vol_regime_20_60"]

    # --- Orderbook features (vectorized if data is simple numeric depth arrays) ---
    if "bids" in df.columns and "asks" in df.columns:
        # Keep this lightweight and safe; if parsing fails, fill zeros.
        def _safe_obi(cell_bids, cell_asks, tau):
            try:
                bids = np.asarray(cell_bids, dtype=float)
                asks = np.asarray(cell_asks, dtype=float)
                if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                    return 0.0
                w = np.exp(-tau * np.arange(min(len(bids), len(asks))))
                bid_vol = np.sum(bids[: len(w), 1] * w)
                ask_vol = np.sum(asks[: len(w), 1] * w)
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
            except Exception:
                return 0.0

        feat["obi_tau1"] = [ _safe_obi(b, a, 1.0) for b, a in zip(df["bids"], df["asks"]) ]
        feat["obi_tau3"] = [ _safe_obi(b, a, 3.0) for b, a in zip(df["bids"], df["asks"]) ]

    # --- Clean up ---
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Ensure feature count stays within limits
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
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
    "min_child_weight": 3,
    "reg_alpha": 0.08,
    "reg_lambda": 0.55,
    "subsample": 0.85,
    "colsample_bytree": 0.82,
    "learning_rate": 0.065,
    "gamma": 0.03,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
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
    eps = 1e-9

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --- Short-horizon returns with rolling normalization ---
    ret_1 = close.pct_change(1)
    feat["ret_1"] = ret_1
    feat["ret_2"] = close.pct_change(2)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_8"] = close.pct_change(8)
    feat["ret_13"] = close.pct_change(13)
    feat["ret_21"] = close.pct_change(21)

    for w in [5, 10, 20]:
        mu = ret_1.rolling(w).mean()
        sd = ret_1.rolling(w).std()
        feat[f"ret_z_{w}"] = (ret_1 - mu) / (sd + eps)

    # --- Volatility + regime change ---
    log_ret = np.log(close / close.shift(1))
    rv_5 = log_ret.rolling(5).std()
    rv_10 = log_ret.rolling(10).std()
    rv_20 = log_ret.rolling(20).std()
    rv_40 = log_ret.rolling(40).std()

    feat["vol_5"] = rv_5
    feat["vol_10"] = rv_10
    feat["vol_20"] = rv_20

    feat["vol_trend_5_10"] = (rv_5 - rv_10) / (rv_10 + eps)
    feat["vol_trend_10_20"] = (rv_10 - rv_20) / (rv_20 + eps)
    feat["vol_trend_20_40"] = (rv_20 - rv_40) / (rv_40 + eps)

    vol_base = rv_20.rolling(120).mean()
    feat["vol_regime"] = rv_20 / (vol_base + eps)
    feat["vol_regime_fast"] = rv_10 / (rv_20 + eps)
    feat["vol_regime_change"] = feat["vol_regime"].diff(3)
    feat["vol_regime_z"] = (feat["vol_regime"] - feat["vol_regime"].rolling(40).mean()) / (
        feat["vol_regime"].rolling(40).std() + eps
    )
    feat["vol_rank_120"] = rv_20.rolling(120).rank(pct=True)

    # --- Trend / efficiency / acceleration ---
    for w in [5, 10, 20]:
        roc = close / close.shift(w) - 1
        feat[f"trend_{w}"] = roc
        feat[f"trend_abs_{w}"] = roc.abs()
        denom = close.diff().abs().rolling(w).sum()
        feat[f"efficiency_{w}"] = (close - close.shift(w)).abs() / (denom + eps)

    feat["efficiency_trend"] = feat["efficiency_5"] - feat["efficiency_20"]
    feat["trend_accel_5"] = feat["trend_5"].diff(2)
    feat["trend_accel_10"] = feat["trend_10"].diff(3)

    # --- Persistence / market character change ---
    feat["ret_autocorr_5"] = ret_1.rolling(5).corr(ret_1.shift(1))
    feat["ret_autocorr_10"] = ret_1.rolling(10).corr(ret_1.shift(1))
    feat["autocorr_trend"] = feat["ret_autocorr_5"] - feat["ret_autocorr_10"]
    feat["autocorr_change"] = feat["autocorr_trend"].diff(3)
    feat["efficiency_change"] = feat["efficiency_5"] - feat["efficiency_10"]
    feat["vol_trend_change"] = feat["vol_trend_10_20"].diff(3)

    # --- Oscillators ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    for period in [5, 9, 14]:
        rs = gain.rolling(period).mean() / (loss.rolling(period).mean() + eps)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # --- Bollinger-style and mean reversion anchors ---
    for w in [10, 20]:
        mid = close.rolling(w).mean()
        std = close.rolling(w).std()
        feat[f"bb_pos_{w}"] = (close - mid) / (std + eps)
        feat[f"bb_width_{w}"] = (2 * std) / (mid + eps)

    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    feat["price_sma5_dev"] = (close - sma_5) / (sma_5 + eps)
    feat["price_sma10_dev"] = (close - sma_10) / (sma_10 + eps)
    feat["price_sma20_dev"] = (close - sma_20) / (sma_20 + eps)
    feat["sma5_sma20_cross"] = (sma_5 - sma_20) / (sma_20 + eps)

    # --- MACD-like short momentum ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd = ema8 - ema21
    feat["macd"] = macd / (close + eps)
    feat["macd_signal"] = (macd - macd.ewm(span=5, adjust=False).mean()) / (close + eps)

    # --- Range / candle structure ---
    hl_range = (high - low) / (close + eps)
    feat["hl_range"] = hl_range
    feat["hl_range_ma5"] = hl_range.rolling(5).mean()
    feat["hl_range_ma20"] = hl_range.rolling(20).mean()
    feat["hl_range_change"] = feat["hl_range_ma5"] - feat["hl_range_ma20"]

    feat["candle_body_ratio"] = (close - open_).abs() / ((high - low) + eps)
    feat["upper_wick_ratio"] = (high - np.maximum(open_, close)) / ((high - low) + eps)
    feat["lower_wick_ratio"] = (np.minimum(open_, close) - low) / ((high - low) + eps)

    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat["close_pos_10"] = (close - low_10) / (high_10 - low_10 + eps)
    feat["close_pos_20"] = (close - low_20) / (high_20 - low_20 + eps)

    # --- Volume / VWAP / flow ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat["volume_ratio_5"] = volume / (vol_ma_5 + eps)
    feat["volume_ratio_20"] = volume / (vol_ma_20 + eps)
    feat["volume_z_20"] = (volume - vol_ma_20) / (vol_std_20 + eps)
    feat["volume_change_5"] = volume.pct_change(5)
    feat["volume_change_20"] = volume.pct_change(20)

    vwap_10 = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + eps)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)
    feat["vwap_dev_10"] = (close - vwap_10) / (close + eps)
    feat["vwap_dev_20"] = (close - vwap_20) / (close + eps)

    feat["vol_ret_corr_20"] = log_ret.rolling(20).corr(volume.pct_change(1))

    # --- Interactions: regime-coupled and short-horizon ---
    feat["momentum_vol_interaction_1"] = feat["ret_3"] * feat["vol_regime"]
    feat["momentum_vol_interaction_2"] = feat["ret_5"] * feat["vol_regime"]
    feat["momentum_regime_ratio_1"] = feat["ret_3"] / (rv_10 + eps)
    feat["momentum_regime_ratio_2"] = feat["ret_5"] / (rv_20 + eps)
    feat["momentum_regime_fast"] = feat["ret_3"] * feat["vol_regime_fast"]
    feat["price_vol_pressure"] = feat["price_sma10_dev"] * feat["vol_regime"]
    feat["volume_price_pressure"] = feat["volume_ratio_20"] * feat["price_sma10_dev"]
    feat["trend_x_vol_change"] = feat["trend_10"] * feat["vol_regime_change"]
    feat["efficiency_x_vol"] = feat["efficiency_10"] * feat["vol_regime"]
    feat["autocorr_x_vol"] = feat["autocorr_trend"] * feat["vol_regime"]

    feat["ret_spread_3_10"] = feat["ret_3"] - close.pct_change(10)
    feat["ret_spread_5_21"] = feat["ret_5"] - feat["ret_21"]
    feat["vol_momentum"] = rv_5 / (rv_20 + eps) - 1.0
    feat["range_x_vol"] = feat["hl_range"] * feat["vol_regime"]
    feat["body_x_range"] = feat["candle_body_ratio"] * feat["hl_range"]
    feat["trend_x_efficiency"] = feat["trend_10"] * feat["efficiency_10"]

    # --- Session / time ---
    if hasattr(df.index, "hour"):
        feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook imbalance if present ---
    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row["bids"])
                asks = np.array(row["asks"])
                if bids.size == 0 or asks.size == 0:
                    return 0.0
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau2"] = df.apply(lambda r: obi(r, 2), axis=1)

    # --- Funding features if present ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_ma_8h"] = fr.rolling(32).mean()
        feat["funding_momentum_4"] = fr.diff(4)
        feat["funding_z_32"] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + eps)
        feat["funding_x_volregime"] = fr * feat["vol_regime"]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Keep feature count comfortably below 80
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
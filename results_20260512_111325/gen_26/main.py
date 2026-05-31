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
    "n_estimators": 360,
    "max_depth": 4,
    "min_child_weight": 3,
    "reg_alpha": 0.06,
    "reg_lambda": 0.9,
    "gamma": 0.02,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 55,
}

CLIP_PERCENTILE = 98
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

    log_ret = np.log(close / close.shift(1))
    ret_1 = close.pct_change(1)
    ret_2 = close.pct_change(2)
    ret_3 = close.pct_change(3)
    ret_5 = close.pct_change(5)
    ret_6 = close.pct_change(6)
    ret_12 = close.pct_change(12)

    # --- short returns (regime friendly) ---
    for lag in [1, 2, 3, 5, 6, 12, 24]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    # --- rolling z-scores of returns ---
    for w in [5, 10, 20]:
        mu = ret_1.rolling(w).mean()
        sd = ret_1.rolling(w).std()
        feat[f"ret_z_{w}"] = (ret_1 - mu) / (sd + 1e-9)

    feat["ret_3_z_20"] = (ret_3 - ret_3.rolling(20).mean()) / (ret_3.rolling(20).std() + 1e-9)
    feat["ret_5_z_20"] = (ret_5 - ret_5.rolling(20).mean()) / (ret_5.rolling(20).std() + 1e-9)
    feat["ret_accel_3"] = ret_1.diff(3)
    feat["ret_accel_6"] = ret_1.diff(6)
    feat["ret_accel_z"] = (feat["ret_accel_3"] - feat["ret_accel_3"].rolling(20).mean()) / (
        feat["ret_accel_3"].rolling(20).std() + 1e-9
    )

    # --- short volatility and volatility regime shifts ---
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()

    feat["vol_5"] = vol_5
    feat["vol_10"] = vol_10
    feat["vol_20"] = vol_20
    feat["vol_10_z"] = (vol_10 - vol_10.rolling(40).mean()) / (vol_10.rolling(40).std() + 1e-9)
    feat["vol_20_z"] = (vol_20 - vol_20.rolling(60).mean()) / (vol_20.rolling(60).std() + 1e-9)
    feat["vol_regime"] = vol_20 / (vol_20.rolling(100).mean() + 1e-9)
    feat["vol_regime_fast"] = vol_10 / (vol_20 + 1e-9)
    feat["vol_trend"] = (vol_10 - vol_10.shift(5)) / (vol_10.shift(5) + 1e-9)
    feat["vol_trend_10"] = (vol_20 - vol_20.shift(10)) / (vol_20.shift(10) + 1e-9)
    feat["vol_rank_100"] = vol_20.rolling(100).rank(pct=True)

    # --- efficiency / character change ---
    eff_10 = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    eff_15 = (close - close.shift(15)).abs() / (close.diff().abs().rolling(15).sum() + 1e-9)
    eff_20 = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat["efficiency_10"] = eff_10
    feat["efficiency_15"] = eff_15
    feat["efficiency_20"] = eff_20
    feat["efficiency_trend"] = eff_10 - eff_20
    feat["efficiency_trend_z"] = (feat["efficiency_trend"] - feat["efficiency_trend"].rolling(20).mean()) / (
        feat["efficiency_trend"].rolling(20).std() + 1e-9
    )
    feat["eff_delta_5"] = eff_10.diff(5) / (eff_10.shift(5) + 1e-9)

    # --- short moving average structure ---
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()

    dev_5 = (close - sma_5) / (sma_5 + 1e-9)
    dev_10 = (close - sma_10) / (sma_10 + 1e-9)
    dev_20 = (close - sma_20) / (sma_20 + 1e-9)

    feat["price_sma5_dev"] = dev_5
    feat["price_sma10_dev"] = dev_10
    feat["price_sma20_dev"] = dev_20
    feat["price_sma10_dev_z"] = (dev_10 - dev_10.rolling(20).mean()) / (dev_10.rolling(20).std() + 1e-9)
    feat["sma_cross_5_20"] = (sma_5 - sma_20) / (sma_20 + 1e-9)
    feat["sma_cross_10_20"] = (sma_10 - sma_20) / (sma_20 + 1e-9)

    # --- oscillators ---
    for period in [5, 10, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # --- range / candle structure ---
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    feat["atr_7"] = tr.rolling(7).mean()
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_norm"] = feat["atr_14"] / (close + 1e-9)
    feat["hl_range"] = (high - low) / (close + 1e-9)
    feat["hl_range_z"] = (feat["hl_range"] - feat["hl_range"].rolling(20).mean()) / (
        feat["hl_range"].rolling(20).std() + 1e-9
    )
    feat["candle_body_ratio"] = (close - open_).abs() / ((high - low) + 1e-9)
    feat["upper_wick_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / ((high - low) + 1e-9)
    feat["lower_wick_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / ((high - low) + 1e-9)

    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat["close_position_20"] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # --- volume normalization ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    feat["volume_ratio_5"] = volume / (vol_ma_5 + 1e-9)
    feat["volume_ratio_20"] = volume / (vol_ma_20 + 1e-9)
    feat["volume_z_20"] = (volume - vol_ma_20) / (volume.rolling(20).std() + 1e-9)
    feat["volume_trend"] = (vol_ma_5 - vol_ma_20) / (vol_ma_20 + 1e-9)
    feat["volume_trend_z"] = (feat["volume_trend"] - feat["volume_trend"].rolling(20).mean()) / (
        feat["volume_trend"].rolling(20).std() + 1e-9
    )

    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat["vwap_dev"] = (close - vwap_20) / (close + 1e-9)

    # --- interactions that can generalize across regimes ---
    feat["momentum_vol_interaction"] = ret_3 * vol_10
    feat["momentum_regime"] = ret_3 / (vol_10 + 1e-9)
    feat["momentum_regime_fast"] = ret_3 * feat["vol_regime_fast"]
    feat["momentum_volregime"] = ret_6 * feat["vol_regime"]
    feat["momentum_x_volregime_z"] = feat["ret_3_z_20"] * (
        (feat["vol_regime"] - feat["vol_regime"].rolling(20).mean()) / (feat["vol_regime"].rolling(20).std() + 1e-9)
    )
    feat["volume_price_pressure"] = feat["volume_ratio_20"] * feat["price_sma10_dev"]
    feat["volume_bb_interaction"] = feat["volume_ratio_20"] * (
        (close - close.rolling(20).mean()) / (2 * close.rolling(20).std() + 1e-9)
    )
    feat["price_x_vol"] = dev_10 * feat["vol_regime_fast"]
    feat["vol_x_effshift"] = feat["vol_trend"] * feat["efficiency_trend"]

    # --- MACD-style short trend ---
    ema6 = close.ewm(span=6, adjust=False).mean()
    ema13 = close.ewm(span=13, adjust=False).mean()
    macd = ema6 - ema13
    feat["macd_signal"] = macd - macd.ewm(span=5, adjust=False).mean()

    # --- time features ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- orderbook (if available) ---
    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row["bids"])
                asks = np.array(row["asks"])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau3"] = df.apply(lambda r: obi(r, 3), axis=1)
        feat["obi_delta"] = feat["obi_tau1"].diff(3)
        feat["obi_x_vol"] = feat["obi_tau1"] * feat["vol_regime_fast"]

    # --- funding rate ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_8h_ma"] = fr.rolling(32).mean()
        feat["funding_momentum"] = fr.diff(4)
        feat["funding_trend"] = fr.diff(8)
        feat["funding_x_volregime"] = fr * feat["vol_regime"]
        feat["funding_x_momentum"] = fr * ret_3

    # clean up
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

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
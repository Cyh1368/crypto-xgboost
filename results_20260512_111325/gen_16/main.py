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
    "n_estimators": 420,
    "max_depth": 4,
    "min_child_weight": 6,
    "reg_alpha": 0.15,
    "reg_lambda": 2.0,
    "gamma": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.78,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 45,
}

CLIP_PERCENTILE = 97
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-transition focused features using only past data.
    Compact set of short-horizon, z-scored, and delta-based signals.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    log_ret = np.log(close / close.shift(1))
    ret1 = close.pct_change(1)
    ret2 = close.pct_change(2)
    ret3 = close.pct_change(3)
    ret6 = close.pct_change(6)
    ret12 = close.pct_change(12)

    # --- Short returns and short rolling z-scores ---
    for lag in [1, 2, 3, 6, 12, 24]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    for w in [5, 10, 20]:
        mu = ret1.rolling(w).mean()
        sd = ret1.rolling(w).std()
        feat[f"ret_z_{w}"] = (ret1 - mu) / (sd + 1e-9)

    # --- Volatility state and volatility change ---
    vol5 = log_ret.rolling(5).std()
    vol10 = log_ret.rolling(10).std()
    vol20 = log_ret.rolling(20).std()

    feat["vol_5"] = vol5
    feat["vol_10"] = vol10
    feat["vol_20"] = vol20
    feat["vol_10_z"] = (vol10 - vol10.rolling(40).mean()) / (vol10.rolling(40).std() + 1e-9)
    feat["vol_20_z"] = (vol20 - vol20.rolling(60).mean()) / (vol20.rolling(60).std() + 1e-9)
    feat["vol_trend_5"] = (vol10 - vol10.shift(5)) / (vol10.shift(5) + 1e-9)
    feat["vol_trend_10"] = (vol20 - vol20.shift(10)) / (vol20.shift(10) + 1e-9)
    feat["vol_regime"] = vol20 / (vol20.rolling(120).mean() + 1e-9)
    feat["vol_regime_fast"] = vol10 / (vol20 + 1e-9)

    # --- Efficiency and its trend ---
    eff10 = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    eff15 = (close - close.shift(15)).abs() / (close.diff().abs().rolling(15).sum() + 1e-9)
    eff20 = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat["eff_10"] = eff10
    feat["eff_15"] = eff15
    feat["eff_20"] = eff20
    feat["eff_trend"] = (eff10 - eff20)
    feat["eff_delta_5"] = eff10.diff(5) / (eff10.shift(5) + 1e-9)

    # --- Price location / short-term trend geometry ---
    sma5 = close.rolling(5).mean()
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()

    dev5 = (close - sma5) / (sma5 + 1e-9)
    dev10 = (close - sma10) / (sma10 + 1e-9)
    dev20 = (close - sma20) / (sma20 + 1e-9)

    feat["price_sma5_dev"] = dev5
    feat["price_sma10_dev"] = dev10
    feat["price_sma20_dev"] = dev20
    feat["price_sma10_dev_z"] = (dev10 - dev10.rolling(20).mean()) / (dev10.rolling(20).std() + 1e-9)
    feat["sma_cross_5_20"] = (sma5 - sma20) / (sma20 + 1e-9)
    feat["sma_cross_10_20"] = (sma10 - sma20) / (sma20 + 1e-9)

    # --- Momentum change / acceleration ---
    feat["mom_3"] = ret3
    feat["mom_6"] = ret6
    feat["mom_12"] = ret12
    feat["mom_accel_3"] = ret1.diff(3)
    feat["mom_accel_6"] = ret1.diff(6)
    feat["mom_accel_z"] = (feat["mom_accel_3"] - feat["mom_accel_3"].rolling(20).mean()) / (
        feat["mom_accel_3"].rolling(20).std() + 1e-9
    )

    # --- Range / candle structure ---
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

    # --- Volume state and volume change ---
    vma5 = volume.rolling(5).mean()
    vma20 = volume.rolling(20).mean()
    feat["volume_ratio_5"] = volume / (vma5 + 1e-9)
    feat["volume_ratio_20"] = volume / (vma20 + 1e-9)
    feat["volume_z_20"] = (volume - vma20) / (volume.rolling(20).std() + 1e-9)
    feat["volume_trend"] = (vma5 - vma20) / (vma20 + 1e-9)
    feat["volume_trend_z"] = (feat["volume_trend"] - feat["volume_trend"].rolling(20).mean()) / (
        feat["volume_trend"].rolling(20).std() + 1e-9
    )

    vwap20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat["vwap_dev"] = (close - vwap20) / (close + 1e-9)

    # --- Oscillators ---
    for period in [5, 10, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    ema6 = close.ewm(span=6, adjust=False).mean()
    ema13 = close.ewm(span=13, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd = ema6 - ema13
    feat["macd_signal"] = macd - macd.ewm(span=5, adjust=False).mean()
    feat["ema_spread_6_21"] = (ema6 - ema21) / (close + 1e-9)

    # --- Change-of-character features ---
    feat["mom_x_volregime"] = ret3 * feat["vol_regime"]
    feat["mom_x_volshift"] = ret3 * feat["vol_trend_5"]
    feat["mom_x_effshift"] = ret3 * feat["eff_trend"]
    feat["vol_x_effshift"] = feat["vol_trend_5"] * feat["eff_trend"]
    feat["price_x_vol"] = dev10 * feat["vol_regime_fast"]
    feat["pressure_x_vol"] = feat["volume_ratio_20"] * dev10

    # --- Directional consistency ---
    feat["up_ratio_10"] = (ret1 > 0).rolling(10).mean()
    feat["up_ratio_20"] = (ret1 > 0).rolling(20).mean()
    feat["ret_sign_consistency_10"] = np.sign(close.diff()).rolling(10).mean()
    feat["ret_sign_consistency_20"] = np.sign(close.diff()).rolling(20).mean()

    # --- Time features ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook ---
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

    # --- Funding rate ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_8h_ma"] = fr.rolling(32).mean()
        feat["funding_momentum"] = fr.diff(4)
        feat["funding_trend"] = fr.diff(8)
        feat["funding_x_volregime"] = fr * feat["vol_regime"]
        feat["funding_x_momentum"] = fr * ret3

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Light guard against excessive dimensionality
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
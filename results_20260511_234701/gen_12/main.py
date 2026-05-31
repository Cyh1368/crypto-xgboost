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
    "n_estimators": 480,
    "max_depth": 4,
    "min_child_weight": 4,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "subsample": 0.78,
    "colsample_bytree": 0.75,
    "learning_rate": 0.045,
    "gamma": 0.01,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 99
MIN_PRED_STD_RATIO = 0.16
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

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    eps = 1e-9
    log_ret = np.log(close / close.shift(1))

    def rz(x, w):
        m = x.rolling(w).mean()
        s = x.rolling(w).std()
        return (x - m) / (s + eps)

    # --- Short-horizon returns and normalized momentum ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        r = close.pct_change(lag)
        feat[f"ret_{lag}"] = r
        feat[f"ret_{lag}_z10"] = rz(r, 10)
        feat[f"ret_{lag}_z20"] = rz(r, 20)

    feat["logret_1"] = log_ret
    feat["logret_1_z20"] = rz(log_ret, 20)

    # --- Short volatility structure / volatility trend ---
    for w in [5, 10, 20]:
        vol = log_ret.rolling(w).std()
        feat[f"vol_{w}"] = vol
        feat[f"vol_{w}_z40"] = rz(vol, 40)

    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    vol_40 = log_ret.rolling(40).std()

    feat["vol_trend_5_20"] = (vol_5 - vol_20) / (vol_20 + eps)
    feat["vol_trend_10_40"] = (vol_10 - vol_40) / (vol_40 + eps)
    feat["vol_ratio_5_20"] = vol_5 / (vol_20 + eps)
    feat["vol_ratio_10_20"] = vol_10 / (vol_20 + eps)
    feat["vol_rank_60"] = vol_20.rolling(60).rank(pct=True)

    # --- Efficiency / autocorrelation proxies ---
    eff_5 = (close - close.shift(5)).abs() / (close.diff().abs().rolling(5).sum() + eps)
    eff_10 = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + eps)
    eff_20 = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + eps)
    feat["eff_5"] = eff_5
    feat["eff_10"] = eff_10
    feat["eff_20"] = eff_20
    feat["eff_trend"] = eff_5 - eff_20
    feat["eff_z20"] = rz(eff_10, 20)

    # --- RSI on short windows, then normalized ---
    for period in [5, 8, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - 100 / (1 + rs)
        feat[f"rsi_{period}"] = rsi
        feat[f"rsi_{period}_z40"] = rz(rsi, 40)

    # --- Price location / mean reversion ---
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()

    feat["price_sma5_dev"] = (close - sma_5) / (sma_5 + eps)
    feat["price_sma10_dev"] = (close - sma_10) / (sma_10 + eps)
    feat["price_sma20_dev"] = (close - sma_20) / (sma_20 + eps)
    feat["sma5_10_cross"] = (sma_5 - sma_10) / (sma_10 + eps)
    feat["sma10_20_cross"] = (sma_10 - sma_20) / (sma_20 + eps)
    feat["ema_diff"] = (ema_8 - ema_21) / (close + eps)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_pct"] = (close - bb_mid) / (2 * bb_std + eps)
    feat["bb_pct_z40"] = rz(feat["bb_pct"], 40)

    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat["close_pos_10"] = (close - low_10) / (high_10 - low_10 + eps)
    feat["close_pos_20"] = (close - low_20) / (high_20 - low_20 + eps)

    # --- ATR / candle structure ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_5 = tr.rolling(5).mean()
    atr_10 = tr.rolling(10).mean()
    feat["atr_5"] = atr_5 / (close + eps)
    feat["atr_10"] = atr_10 / (close + eps)
    feat["atr_5_z40"] = rz(feat["atr_5"], 40)
    feat["atr_10_z40"] = rz(feat["atr_10"], 40)

    feat["hl_range"] = (high - low) / (close + eps)
    feat["hl_range_z20"] = rz(feat["hl_range"], 20)
    feat["candle_body_ratio"] = (close - open_).abs() / ((high - low) + eps)
    feat["upper_wick_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / ((high - low) + eps)
    feat["lower_wick_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / ((high - low) + eps)

    # --- Volume and volume regime ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_20 = volume.rolling(20).mean()
    feat["volume_ratio_5"] = volume / (vol_ma_5 + eps)
    feat["volume_ratio_10"] = volume / (vol_ma_10 + eps)
    feat["volume_ratio_20"] = volume / (vol_ma_20 + eps)
    feat["volume_z20"] = rz(volume, 20)
    feat["volume_z40"] = rz(volume, 40)

    vwap_10 = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + eps)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)
    feat["vwap_dev_10"] = (close - vwap_10) / (close + eps)
    feat["vwap_dev_20"] = (close - vwap_20) / (close + eps)

    # --- Change-of-market-character features ---
    feat["vol_trend_z"] = rz(feat["vol_trend_5_20"], 40)
    feat["eff_trend_z"] = rz(feat["eff_trend"], 40)
    feat["ret_accel_3"] = close.pct_change(1).diff(3)
    feat["ret_accel_3_z20"] = rz(feat["ret_accel_3"], 20)

    # --- Regime features ---
    trend_5 = close / close.shift(5) - 1
    trend_10 = close / close.shift(10) - 1
    trend_20 = close / close.shift(20) - 1

    feat["trend_dir_5"] = np.sign(trend_5)
    feat["trend_dir_10"] = np.sign(trend_10)
    feat["trend_strength_5"] = trend_5.abs() * 10000
    feat["trend_strength_10"] = trend_10.abs() * 10000
    feat["trend_strength_20"] = trend_20.abs() * 10000

    feat["trend_strength_5_z40"] = rz(feat["trend_strength_5"], 40)
    feat["trend_strength_20_z40"] = rz(feat["trend_strength_20"], 40)

    feat["vol_regime"] = vol_20 / (vol_20.rolling(80).mean() + eps)
    feat["vol_regime_fast"] = vol_10 / (vol_20 + eps)
    feat["vol_regime_trend"] = (vol_10 - vol_40) / (vol_40 + eps)
    feat["vol_regime_z40"] = rz(feat["vol_regime"], 40)

    # --- Volatility-of-Volatility: capture regime shifts ---
    # Second-order changes in volatility signal when the market regime is unstable
    vol_5_roll = vol_5.rolling(10)
    vol_10_roll = vol_10.rolling(10)
    vol_20_roll = vol_20.rolling(10)

    vol_of_vol_5 = vol_5_roll.std()
    vol_of_vol_10 = vol_10_roll.std()
    vol_of_vol_20 = vol_20_roll.std()

    feat["vol_of_vol_5"] = vol_of_vol_5 / (vol_5.rolling(40).mean() + eps)
    feat["vol_of_vol_10"] = vol_of_vol_10 / (vol_10.rolling(40).mean() + eps)
    feat["vol_of_vol_20"] = vol_of_vol_20 / (vol_20.rolling(40).mean() + eps)

    # Rank-based regime change detection (smoother than absolute values)
    feat["vol_of_vol_5_rank"] = vol_of_vol_5.rolling(60).rank(pct=True)
    feat["vol_of_vol_10_rank"] = vol_of_vol_10.rolling(60).rank(pct=True)

    # Composite volatility regime indicator
    feat["vol_regime_instability"] = (vol_of_vol_5 + vol_of_vol_10) / (vol_of_vol_20 + eps)

    # --- Interactions that often generalize better ---
    ret_3 = close.pct_change(3)
    feat["mom_x_volreg"] = ret_3 * feat["vol_regime"]
    feat["mom_x_voltrend"] = ret_3 * feat["vol_regime_trend"]
    feat["mom_x_eff"] = ret_3 * feat["eff_10"]
    feat["mom_x_efftrend"] = ret_3 * feat["eff_trend"]
    feat["vol_x_eff"] = feat["vol_regime"] * feat["eff_10"]
    feat["price_x_vol"] = feat["price_sma10_dev"] * feat["vol_regime"]
    feat["bb_x_vol"] = feat["bb_pct"] * feat["vol_regime"]

    # Regime-instability weighted interactions
    feat["mom_x_volinst"] = ret_3 * feat["vol_regime_instability"]
    feat["eff_x_volinst"] = feat["eff_10"] * feat["vol_regime_instability"]
    feat["trend_x_volinst"] = trend_10 * feat["vol_regime_instability"]

    # --- Session / time ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook (if present) ---
    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row["bids"])
                asks = np.array(row["asks"])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau3"] = df.apply(lambda r: obi(r, 3), axis=1)
        feat["obi_tau1_z40"] = rz(feat["obi_tau1"], 40)

    # --- Funding rate (if present) ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"]
        feat["funding_rate"] = fr
        feat["funding_8h_ma"] = fr.rolling(32).mean()
        feat["funding_momentum"] = fr.diff(4)
        feat["funding_z40"] = rz(fr, 40)
        feat["funding_x_volregime"] = fr * feat["vol_regime"]
        feat["funding_x_volinst"] = fr * feat["vol_regime_instability"]

    # --- Adaptive volatility normalization (regime-invariant) ---
    # Normalize raw features by realized volatility to make them regime-agnostic
    realized_vol = log_ret.rolling(20).std()

    # Normalize return-based features by recent volatility
    for lag in [1, 3, 5]:
        if f"ret_{lag}" in feat.columns:
            feat[f"ret_{lag}_vola"] = feat[f"ret_{lag}"] / (realized_vol + eps)

    # Normalize price deviation features by volatility
    feat["price_sma5_dev_vola"] = feat["price_sma5_dev"] / (realized_vol.rolling(20).mean() + eps)
    feat["price_sma10_dev_vola"] = feat["price_sma10_dev"] / (realized_vol.rolling(20).mean() + eps)
    feat["bb_pct_vola"] = feat["bb_pct"] / (realized_vol.rolling(20).mean() + eps)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
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
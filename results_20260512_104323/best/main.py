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
    "reg_alpha": 0.35,
    "reg_lambda": 2.0,
    "gamma": 0.15,
    "subsample": 0.75,
    "colsample_bytree": 0.70,
    "learning_rate": 0.055,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 35,
}

CLIP_PERCENTILE = 95
MIN_PRED_STD_RATIO = 0.12
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

    close = df['close'].astype(float)
    open_ = df['open'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    log_close = np.log(close.replace(0, np.nan))
    log_ret = log_close.diff()
    ret_1 = close.pct_change(1)
    ret_3 = close.pct_change(3)
    ret_6 = close.pct_change(6)

    # --- Short-horizon returns and rolling z-scores ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    for w in [5, 10, 20]:
        m = ret_1.rolling(w).mean()
        s = ret_1.rolling(w).std()
        feat[f"ret_z_{w}"] = (ret_1 - m) / (s + 1e-9)

    # --- Short volatility and volatility regime change ---
    for w in [5, 10, 20]:
        rv = log_ret.rolling(w).std()
        feat[f"vol_{w}"] = rv
        feat[f"vol_z_{w}"] = (rv - rv.rolling(50).mean()) / (rv.rolling(50).std() + 1e-9)

    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat["vol_regime"] = vol_10 / (vol_20 + 1e-9)
    feat["vol_regime_trend"] = feat["vol_regime"] - feat["vol_regime"].rolling(10).mean()
    feat["vol_accel"] = vol_10.diff(3)

    # --- Trend / momentum with shorter decay ---
    ema_5 = close.ewm(span=5, adjust=False).mean()
    ema_13 = close.ewm(span=13, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    feat["ema_gap_5_13"] = (ema_5 - ema_13) / (close + 1e-9)
    feat["ema_gap_13_21"] = (ema_13 - ema_21) / (close + 1e-9)

    feat["mom_3"] = ret_3
    feat["mom_6"] = ret_6
    feat["mom_3_z20"] = (ret_3 - ret_3.rolling(20).mean()) / (ret_3.rolling(20).std() + 1e-9)
    feat["mom_6_z20"] = (ret_6 - ret_6.rolling(20).mean()) / (ret_6.rolling(20).std() + 1e-9)

    # --- RSI / efficiency / autocorrelation ---
    for period in [5, 9, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    feat["efficiency_10"] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat["efficiency_20"] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat["eff_trend"] = feat["efficiency_10"] - feat["efficiency_20"]

    feat["autocorr_5"] = ret_1.rolling(10).corr(ret_1.shift(1))
    feat["autocorr_10"] = ret_1.rolling(20).corr(ret_1.shift(1))
    feat["autocorr_trend"] = feat["autocorr_5"] - feat["autocorr_10"]

    # --- Mean reversion / price position ---
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    feat["price_sma5_dev"] = (close - sma_5) / (sma_5 + 1e-9)
    feat["price_sma10_dev"] = (close - sma_10) / (sma_10 + 1e-9)
    feat["price_sma20_dev"] = (close - sma_20) / (sma_20 + 1e-9)
    feat["sma5_20_gap"] = (sma_5 - sma_20) / (sma_20 + 1e-9)

    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    feat["close_position_10"] = (close - low_10) / (high_10 - low_10 + 1e-9)

    # --- Candle structure / range ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["atr_5"] = tr.rolling(5).mean()
    feat["atr_10"] = tr.rolling(10).mean()
    feat["atr_norm_10"] = feat["atr_10"] / (close + 1e-9)
    feat["hl_range"] = (high - low) / (close + 1e-9)
    feat["hl_range_z20"] = (feat["hl_range"] - feat["hl_range"].rolling(20).mean()) / (feat["hl_range"].rolling(20).std() + 1e-9)
    feat["body_ratio"] = (close - open_).abs() / ((high - low) + 1e-9)
    feat["upper_wick_ratio"] = (high - np.maximum(open_, close)) / ((high - low) + 1e-9)
    feat["lower_wick_ratio"] = (np.minimum(open_, close) - low) / ((high - low) + 1e-9)

    # --- Volume and volume surprise ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat["volume_ratio_5"] = volume / (vol_ma_5 + 1e-9)
    feat["volume_ratio_20"] = volume / (vol_ma_20 + 1e-9)
    feat["volume_z_20"] = (volume - vol_ma_20) / (vol_std_20 + 1e-9)
    feat["volume_trend"] = feat["volume_ratio_5"] - feat["volume_ratio_20"]

    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat["vwap_dev"] = (close - vwap_20) / (close + 1e-9)

    # --- Regime interactions ---
    feat["mom_x_volreg"] = ret_3 * feat["vol_regime"]
    feat["mom_x_eff"] = ret_3 * feat["efficiency_10"]
    feat["vol_x_efftrend"] = feat["vol_regime"] * feat["eff_trend"]
    feat["vol_x_autocorr"] = feat["vol_regime"] * feat["autocorr_trend"]
    feat["vol_x_momz"] = feat["vol_regime"] * feat["mom_3_z20"]
    feat["volume_x_volreg"] = feat["volume_ratio_20"] * feat["vol_regime"]

    # --- Time features ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook (if real data available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau3"] = df.apply(lambda r: obi(r, 3), axis=1)
        feat["obi_x_volreg"] = feat["obi_tau1"] * feat["vol_regime"]

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].astype(float)
        feat["funding_rate"] = fr
        feat["funding_4h_ma"] = fr.rolling(16).mean()
        feat["funding_8h_ma"] = fr.rolling(32).mean()
        feat["funding_momentum"] = fr.diff(4)
        feat["funding_z_32"] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)
        feat["funding_x_volregime"] = fr * feat["vol_regime"]

    # --- Price pressure / range context ---
    feat["ret_accel_3"] = ret_1.diff(3)
    feat["ret_accel_6"] = ret_1.diff(6)
    feat["range_pressure"] = feat["hl_range"] * feat["volume_ratio_20"]
    feat["range_x_mom"] = feat["hl_range"] * ret_3
    feat["price_pressure"] = feat["price_sma10_dev"] * feat["volume_ratio_20"]

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
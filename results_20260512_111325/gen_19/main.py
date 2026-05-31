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
    "n_estimators": 320,
    "max_depth": 4,
    "min_child_weight": 2,
    "reg_alpha": 0.10,
    "reg_lambda": 1.20,
    "gamma": 0.00,
    "subsample": 0.85,
    "colsample_bytree": 0.78,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 45,
}

CLIP_PERCENTILE = 96
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

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    log_ret = np.log(close / close.shift(1))
    ret1 = close.pct_change(1)

    # --- Short-horizon returns ---
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    # Rolling z-score normalization of recent returns
    for w in [10, 20, 40]:
        mu = ret1.rolling(w).mean()
        sd = ret1.rolling(w).std()
        feat[f"ret_z_{w}"] = (ret1 - mu) / (sd + 1e-9)

    # --- Volatility / regime ---
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    vol_40 = log_ret.rolling(40).std()

    feat["vol_5"] = vol_5
    feat["vol_10"] = vol_10
    feat["vol_20"] = vol_20
    feat["vol_40"] = vol_40
    feat["vol_trend_5_20"] = (vol_5 - vol_20) / (vol_20 + 1e-9)
    feat["vol_trend_10_20"] = (vol_10 - vol_20) / (vol_20 + 1e-9)
    feat["vol_trend_20_40"] = (vol_20 - vol_40) / (vol_40 + 1e-9)

    realized_vol_20 = vol_20
    realized_vol_40 = vol_40
    feat["vol_regime"] = realized_vol_20 / (realized_vol_20.rolling(80).mean() + 1e-9)
    feat["vol_regime_fast"] = realized_vol_20 / (realized_vol_40 + 1e-9)
    feat["vol_rank_80"] = realized_vol_20.rolling(80).rank(pct=True)
    feat["vol_regime_change"] = feat["vol_regime"] - feat["vol_regime"].shift(5)

    # --- RSI on short windows ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    for period in [5, 9, 14]:
        rs = gain.rolling(period).mean() / (loss.rolling(period).mean() + 1e-9)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    feat["rsi_z_20"] = (feat["rsi_9"] - feat["rsi_9"].rolling(20).mean()) / (
        feat["rsi_9"].rolling(20).std() + 1e-9
    )

    # --- Bollinger / position ---
    bb_mid_20 = close.rolling(20).mean()
    bb_std_20 = close.rolling(20).std()
    feat["bb_pct"] = (close - bb_mid_20) / (2 * bb_std_20 + 1e-9)
    feat["bb_pct_z_20"] = (feat["bb_pct"] - feat["bb_pct"].rolling(20).mean()) / (
        feat["bb_pct"].rolling(20).std() + 1e-9
    )

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_30 = close.rolling(30).mean()
    feat["price_sma10_dev"] = (close - sma_10) / (sma_10 + 1e-9)
    feat["price_sma20_dev_z"] = (
        ((close - sma_20) / (sma_20 + 1e-9))
        - ((close - sma_20) / (sma_20 + 1e-9)).rolling(40).mean()
    ) / ((((close - sma_20) / (sma_20 + 1e-9)).rolling(40).std()) + 1e-9)
    feat["price_sma30_dev"] = (close - sma_30) / (sma_30 + 1e-9)
    feat["sma_cross_10_30"] = (sma_10 - sma_30) / (sma_30 + 1e-9)

    # --- ATR / candle structure ---
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    feat["atr_10"] = tr.rolling(10).mean()
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_norm"] = feat["atr_10"] / (close + 1e-9)
    feat["atr_trend"] = feat["atr_10"] / (tr.rolling(30).mean() + 1e-9) - 1

    feat["hl_range"] = (high - low) / (close + 1e-9)
    feat["hl_range_ma"] = feat["hl_range"].rolling(10).mean()
    feat["body_ratio"] = (close - open_).abs() / ((high - low) + 1e-9)
    feat["upper_wick_ratio"] = (high - df[["open", "close"]].max(axis=1)) / ((high - low) + 1e-9)
    feat["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Efficiency / autocorrelation trends ---
    feat["efficiency_10"] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat["efficiency_20"] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat["efficiency_trend"] = feat["efficiency_10"] - feat["efficiency_20"]
    feat["ret_autocorr_10"] = ret1.rolling(10).corr(ret1.shift(1))
    feat["ret_autocorr_20"] = ret1.rolling(20).corr(ret1.shift(1))
    feat["autocorr_trend"] = feat["ret_autocorr_10"] - feat["ret_autocorr_20"]
    feat["autocorr_change"] = feat["ret_autocorr_10"] - feat["ret_autocorr_10"].shift(5)

    # --- Volume normalization / local regime ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat["volume_z_20"] = (volume - vol_ma_20) / (vol_std_20 + 1e-9)
    feat["volume_ratio_5"] = volume / (vol_ma_5 + 1e-9)
    feat["volume_ratio_10"] = volume / (vol_ma_10 + 1e-9)
    feat["volume_ratio_20"] = volume / (vol_ma_20 + 1e-9)

    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat["vwap_dev"] = (close - vwap_20) / (close + 1e-9)

    # --- Regime-conditional interactions ---
    ret_3 = close.pct_change(3)
    ret_5 = close.pct_change(5)
    feat["momentum_vol_interaction"] = ret_3 * vol_20
    feat["momentum_regime"] = ret_3 / (vol_20 + 1e-9)
    feat["momentum_regime_20"] = ret_3 * feat["vol_regime"]
    feat["momentum_regime_fast"] = ret_3 * feat["vol_regime_fast"]
    feat["mom_vol_short"] = ret_5 * feat["vol_regime_change"]
    feat["efficiency_vol_interaction"] = feat["efficiency_trend"] * feat["vol_regime"]
    feat["vol_mom_interaction"] = feat["vol_regime"] * feat["ret_1_z_20"]

    feat["volume_bb_interaction"] = feat["volume_ratio_10"] * feat["bb_pct"]
    feat["volume_price_pressure"] = feat["volume_ratio_10"] * feat["price_sma10_dev"]

    # --- Close position in range ---
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat["close_position_20"] = (close - low_20) / (high_20 - low_20 + 1e-9)
    feat["ret_z_20"] = (ret1 - ret1.rolling(20).mean()) / (ret1.rolling(20).std() + 1e-9)
    feat["ret_z_60"] = (ret1 - ret1.rolling(60).mean()) / (ret1.rolling(60).std() + 1e-9)
    feat["ret_accel_6"] = ret1.diff(6)
    feat["vol_ret_corr_20"] = log_ret.rolling(20).corr(volume.pct_change(1))

    # --- Trend strength / direction ---
    feat["trend_strength_10"] = (close / close.shift(10) - 1).abs() * 10000
    feat["trend_strength_20"] = (close / close.shift(20) - 1).abs() * 10000
    feat["trend_dir_10"] = np.sign(close / close.shift(10) - 1)
    feat["trend_dir_20"] = np.sign(close / close.shift(20) - 1)

    # --- Session / Time ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook (if real data available) ---
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

    # --- Funding rate ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"]
        feat["funding_rate"] = fr
        feat["funding_8h_ma"] = fr.rolling(32).mean()
        feat["funding_momentum"] = fr.diff(4)
        feat["funding_x_volregime"] = fr * feat["vol_regime"]

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
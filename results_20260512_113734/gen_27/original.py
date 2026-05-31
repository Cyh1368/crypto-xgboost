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
    "min_child_weight": 10,
    "reg_alpha": 1.5,
    "reg_lambda": 8.0,
    "gamma": 0.2,
    "subsample": 0.72,
    "colsample_bytree": 0.72,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 35,
}

CLIP_PERCENTILE = 95
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features use only past data.
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
    ret1 = close.pct_change(1)
    ret2 = close.pct_change(2)
    ret3 = close.pct_change(3)

    eps = 1e-9

    def zscore(s, w):
        m = s.rolling(w).mean()
        sd = s.rolling(w).std()
        return (s - m) / (sd + eps)

    def rank_pct(s, w):
        return s.rolling(w).rank(pct=True)

    # ── Short-memory returns / momentum
    for lag in [1, 2, 3, 5, 8, 13, 20]:
        feat[f"ret_{lag}"] = close.pct_change(lag)

    feat["ret_1_z_20"] = zscore(ret1, 20)
    feat["ret_3_z_20"] = zscore(ret3, 20)
    feat["ret_1_z_60"] = zscore(ret1, 60)
    feat["ret_3_z_60"] = zscore(ret3, 60)

    # momentum decay / acceleration
    feat["mom_5_20"] = close.pct_change(5) - close.pct_change(20)
    feat["mom_accel_3"] = ret1.diff(3)
    feat["mom_accel_6"] = ret1.diff(6)

    # ── Volatility regime and volatility trends
    for w in [5, 10, 20]:
        feat[f"vol_{w}"] = log_ret.rolling(w).std()

    feat["vol_5_z_60"] = zscore(feat["vol_5"], 60)
    feat["vol_10_z_60"] = zscore(feat["vol_10"], 60)
    feat["vol_20_z_120"] = zscore(feat["vol_20"], 120)
    feat["vol_trend_20"] = feat["vol_20"].diff(5)
    feat["vol_trend_40"] = feat["vol_20"].diff(10)
    feat["vol_rank_120"] = rank_pct(feat["vol_20"], 120)

    # ── Directional efficiency / trend quality
    net_10 = (close - close.shift(10)).abs()
    path_10 = close.diff().abs().rolling(10).sum()
    feat["eff_10"] = net_10 / (path_10 + eps)

    net_20 = (close - close.shift(20)).abs()
    path_20 = close.diff().abs().rolling(20).sum()
    feat["eff_20"] = net_20 / (path_20 + eps)

    feat["eff_trend"] = feat["eff_10"] - feat["eff_20"]
    feat["trend_dir_10"] = np.sign(close.pct_change(10))
    feat["trend_dir_20"] = np.sign(close.pct_change(20))

    # ── Rolling autocorrelation / market memory
    feat["autocorr_5_20"] = ret1.rolling(20).corr(ret1.shift(1))
    feat["autocorr_5_60"] = ret1.rolling(60).corr(ret1.shift(1))
    feat["autocorr_trend"] = feat["autocorr_5_20"] - feat["autocorr_5_60"]

    # ── Rolling standardized price position
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_40 = close.rolling(40).mean()
    std_20 = close.rolling(20).std()
    std_40 = close.rolling(40).std()

    feat["price_z_20"] = (close - sma_20) / (std_20 + eps)
    feat["price_z_40"] = (close - sma_40) / (std_40 + eps)
    feat["sma10_sma20"] = (sma_10 - sma_20) / (sma_20 + eps)
    feat["sma20_sma40"] = (sma_20 - sma_40) / (sma_40 + eps)

    # ── Range / candle structure, normalized
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    feat["atr_10"] = tr.rolling(10).mean()
    feat["atr_20"] = tr.rolling(20).mean()
    feat["atr_norm_10"] = feat["atr_10"] / (close + eps)
    feat["atr_norm_20"] = feat["atr_20"] / (close + eps)
    feat["range_z_20"] = zscore((high - low) / (close + eps), 20)

    candle_range = (high - low).replace(0, np.nan)
    feat["body_ratio"] = (close - open_).abs() / (candle_range + eps)
    feat["upper_wick"] = (high - np.maximum(open_, close)) / (candle_range + eps)
    feat["lower_wick"] = (np.minimum(open_, close) - low) / (candle_range + eps)

    # ── Volume regime normalized
    vol_mean_10 = volume.rolling(10).mean()
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()

    feat["vol_ratio_10"] = volume / (vol_mean_10 + eps)
    feat["vol_ratio_20"] = volume / (vol_mean_20 + eps)
    feat["vol_z_20"] = (volume - vol_mean_20) / (vol_std_20 + eps)
    feat["vol_z_60"] = zscore(volume, 60)
    feat["vol_trend"] = feat["vol_ratio_10"] - feat["vol_ratio_20"]

    # ── Price-volume interactions
    feat["mom_x_volreg"] = ret3 * feat["vol_z_60"]
    feat["mom_x_atrreg"] = ret3 * feat["atr_norm_20"]
    feat["eff_x_volreg"] = feat["eff_10"] * feat["vol_rank_120"]
    feat["range_x_volume"] = feat["range_z_20"] * feat["vol_z_20"]

    # ── Mean reversion / breakout pressure
    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()

    feat["close_pos_10"] = (close - low_10) / (high_10 - low_10 + eps)
    feat["close_pos_20"] = (close - low_20) / (high_20 - low_20 + eps)
    feat["breakout_10"] = ((close > high_10.shift(1)).astype(float) - (close < low_10.shift(1)).astype(float))
    feat["breakout_20"] = ((close > high_20.shift(1)).astype(float) - (close < low_20.shift(1)).astype(float))

    feat["mr_pressure_10"] = 1.0 - feat["close_pos_10"]
    feat["mr_pressure_20"] = 1.0 - feat["close_pos_20"]

    # ── Short-window RSI-like oscillator with normalization
    for period in [5, 9, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - 100 / (1 + rs)
        feat[f"rsi_{period}"] = rsi
        feat[f"rsi_{period}_z_60"] = zscore(rsi, 60)

    # ── Distance from anchored rolling VWAP
    vwap_10 = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + eps)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)
    feat["vwap_dev_10"] = (close - vwap_10) / (close + eps)
    feat["vwap_dev_20"] = (close - vwap_20) / (close + eps)
    feat["vwap_dev_x_vol"] = feat["vwap_dev_20"] * feat["vol_z_20"]

    # ── Session effects
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # ── Funding / orderbook if present
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_z_96"] = zscore(fr, 96)
        feat["funding_mom_4"] = fr.diff(4)
        feat["funding_mom_12"] = fr.diff(12)
        feat["funding_x_mom"] = feat["funding_z_96"] * ret3

    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row["bids"], dtype=float)
                asks = np.array(row["asks"], dtype=float)
                if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                    return 0.0
                b = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b - a) / (b + a + eps)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau3"] = df.apply(lambda r: obi(r, 3), axis=1)
        feat["obi_diff"] = feat["obi_tau1"] - feat["obi_tau3"]

    # ── Stability / regime shift indicators
    feat["return_dispersion_20"] = ret1.rolling(20).std() / (ret1.rolling(20).mean().abs() + eps)
    feat["return_dispersion_60"] = ret1.rolling(60).std() / (ret1.rolling(60).mean().abs() + eps)
    feat["dispersion_trend"] = feat["return_dispersion_20"] - feat["return_dispersion_60"]

    feat["range_to_vol_20"] = feat["range_z_20"] / (feat["vol_z_20"].abs() + 1.0)
    feat["momentum_regime"] = ret3 / (feat["vol_20"] + eps)
    feat["momentum_regime_x_eff"] = feat["momentum_regime"] * feat["eff_10"]

    # Cleanup
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # Keep feature count safely under 80
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
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
    "n_estimators": 220,
    "max_depth": 4,
    "min_child_weight": 6,
    "reg_alpha": 0.8,
    "reg_lambda": 3.5,
    "gamma": 0.15,
    "subsample": 0.75,
    "colsample_bytree": 0.72,
    "learning_rate": 0.06,
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
def _safe_div(a, b, eps=1e-9):
    return a / (b + eps)


def _rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / (sd + 1e-9)


def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _realized_vol(log_ret: pd.Series, win: int) -> pd.Series:
    return log_ret.rolling(win).std()


def _price_momentum(close: pd.Series, lags) -> pd.DataFrame:
    out = {}
    for lag in lags:
        out[f"ret_{lag}"] = close.pct_change(lag)
    return pd.DataFrame(out, index=close.index)


def _regime_features(close: pd.Series, volume: pd.Series, log_ret: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame(index=close.index)

    rv_6 = _realized_vol(log_ret, 6)
    rv_12 = _realized_vol(log_ret, 12)
    rv_24 = _realized_vol(log_ret, 24)
    rv_48 = _realized_vol(log_ret, 48)

    feat["vol_6"] = rv_6
    feat["vol_12"] = rv_12
    feat["vol_24"] = rv_24
    feat["vol_48"] = rv_48

    feat["vol_trend_12_24"] = _safe_div(rv_12 - rv_24, rv_24)
    feat["vol_trend_24_48"] = _safe_div(rv_24 - rv_48, rv_48)
    feat["vol_regime_fast"] = _safe_div(rv_12, rv_48)
    feat["vol_rank_96"] = rv_24.rolling(96).rank(pct=True)
    feat["vol_z_96"] = _rolling_zscore(rv_24, 96)

    # efficiency: direct move vs path length, short horizon
    eff_8 = _safe_div((close - close.shift(8)).abs(), close.diff().abs().rolling(8).sum())
    eff_16 = _safe_div((close - close.shift(16)).abs(), close.diff().abs().rolling(16).sum())
    feat["eff_8"] = eff_8
    feat["eff_16"] = eff_16
    feat["eff_trend"] = _safe_div(eff_8 - eff_16, eff_16)

    # short-horizon autocorrelation proxy (regime-sensitive)
    ret1 = close.pct_change(1)
    feat["ret_acf_6"] = ret1.rolling(6).corr(ret1.shift(1))
    feat["ret_acf_12"] = ret1.rolling(12).corr(ret1.shift(1))
    feat["acf_trend"] = feat["ret_acf_6"] - feat["ret_acf_12"]

    # volume regime and its change
    vol_z = _rolling_zscore(volume, 24)
    feat["volume_z_24"] = vol_z
    feat["volume_z_96"] = _rolling_zscore(volume, 96)
    feat["volume_trend"] = vol_z - feat["volume_z_96"]

    return feat


def _price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    log_ret = _log_return(close)

    # shorter momentum windows
    feat = pd.concat([feat, _price_momentum(close, [1, 2, 3, 4, 6, 8, 12, 16, 24])], axis=1)

    # normalized momentum; better across regimes
    for w in [6, 12, 24]:
        feat[f"ret_z_{w}"] = _rolling_zscore(close.pct_change(1), w)

    # short volatility and vol-of-vol
    feat["volofvol_12"] = _realized_vol(log_ret, 12).rolling(12).std()
    feat["volofvol_24"] = _realized_vol(log_ret, 24).rolling(24).std()

    # candle structure
    hl = (high - low)
    feat["hl_range"] = _safe_div(hl, close)
    feat["body_ratio"] = _safe_div((close - open_).abs(), hl)
    feat["upper_wick"] = _safe_div(high - np.maximum(open_, close), hl)
    feat["lower_wick"] = _safe_div(np.minimum(open_, close) - low, hl)

    # local location in short range
    low_8 = low.rolling(8).min()
    high_8 = high.rolling(8).max()
    low_16 = low.rolling(16).min()
    high_16 = high.rolling(16).max()
    feat["close_pos_8"] = _safe_div(close - low_8, high_8 - low_8)
    feat["close_pos_16"] = _safe_div(close - low_16, high_16 - low_16)

    # deviation from short mean, then z-scored
    sma_8 = close.rolling(8).mean()
    sma_16 = close.rolling(16).mean()
    sma_32 = close.rolling(32).mean()
    feat["dev_sma8"] = _safe_div(close - sma_8, sma_8)
    feat["dev_sma16"] = _safe_div(close - sma_16, sma_16)
    feat["dev_sma32"] = _safe_div(close - sma_32, sma_32)
    feat["dev_sma8_z"] = _rolling_zscore(feat["dev_sma8"], 24)
    feat["dev_sma16_z"] = _rolling_zscore(feat["dev_sma16"], 48)

    # ATR-like short range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_8 = tr.rolling(8).mean()
    atr_16 = tr.rolling(16).mean()
    feat["atr_8"] = _safe_div(atr_8, close)
    feat["atr_16"] = _safe_div(atr_16, close)
    feat["atr_trend"] = _safe_div(atr_8 - atr_16, atr_16)

    # short MACD-style feature, normalized
    ema6 = close.ewm(span=6, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema24 = close.ewm(span=24, adjust=False).mean()
    macd_fast = ema6 - ema12
    macd_slow = ema12 - ema24
    feat["macd_fast"] = _safe_div(macd_fast, close)
    feat["macd_slow"] = _safe_div(macd_slow, close)
    feat["macd_trend"] = feat["macd_fast"] - feat["macd_slow"]

    # volume pressure and price-volume coupling
    vol_ratio_8 = _safe_div(volume, volume.rolling(8).mean())
    vol_ratio_24 = _safe_div(volume, volume.rolling(24).mean())
    feat["vol_ratio_8"] = vol_ratio_8
    feat["vol_ratio_24"] = vol_ratio_24
    feat["vol_ratio_trend"] = vol_ratio_8 - vol_ratio_24
    feat["vol_x_ret3"] = vol_ratio_8 * close.pct_change(3)
    feat["vol_x_dev8"] = vol_ratio_8 * feat["dev_sma8"]

    # rolling correlation (short and robust)
    feat["vol_ret_corr_12"] = log_ret.rolling(12).corr(volume.pct_change(1))
    feat["vol_ret_corr_24"] = log_ret.rolling(24).corr(volume.pct_change(1))

    return feat


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    feat = pd.DataFrame(index=idx)
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    return feat


def _orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if "bids" not in df.columns or "asks" not in df.columns:
        return feat

    def obi(row, tau=1.0):
        try:
            bids = np.asarray(row["bids"], dtype=float)
            asks = np.asarray(row["asks"], dtype=float)
            if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                return 0.0
            wb = np.exp(-tau * np.arange(len(bids)))
            wa = np.exp(-tau * np.arange(len(asks)))
            bid_vol = np.sum(bids[:, 1] * wb)
            ask_vol = np.sum(asks[:, 1] * wa)
            return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        except Exception:
            return 0.0

    feat["obi_tau1"] = df.apply(lambda r: obi(r, 1.0), axis=1)
    feat["obi_tau2"] = df.apply(lambda r: obi(r, 2.0), axis=1)
    feat["obi_tau4"] = df.apply(lambda r: obi(r, 4.0), axis=1)
    return feat


def _funding_features(df: pd.DataFrame, vol_regime: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if "funding_rate" not in df.columns:
        return feat

    fr = df["funding_rate"]
    feat["funding_rate"] = fr
    feat["funding_ma_16"] = fr.rolling(16).mean()
    feat["funding_ma_32"] = fr.rolling(32).mean()
    feat["funding_chg_4"] = fr.diff(4)
    feat["funding_chg_8"] = fr.diff(8)
    feat["funding_z_32"] = _rolling_zscore(fr, 32)
    feat["funding_x_volregime"] = fr * vol_regime
    return feat


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features use only past data.
    """
    close = df["close"]
    volume = df["volume"]
    log_ret = _log_return(close)

    price_feat = _price_action_features(df)
    regime_feat = _regime_features(close, volume, log_ret)
    time_feat = _time_features(df)
    ob_feat = _orderbook_features(df)
    funding_feat = _funding_features(df, regime_feat.get("vol_regime_fast", pd.Series(index=df.index, dtype=float)))

    feat = pd.concat([price_feat, regime_feat, time_feat, ob_feat, funding_feat], axis=1)

    # A few compact regime interactions that often transfer better than raw levels
    if "ret_3" in feat.columns and "vol_regime_fast" in feat.columns:
        feat["mom_x_volreg"] = feat["ret_3"] * feat["vol_regime_fast"]
    if "ret_6" in feat.columns and "eff_8" in feat.columns:
        feat["mom6_x_eff"] = feat["ret_6"] * feat["eff_8"]
    if "dev_sma8" in feat.columns and "volume_z_24" in feat.columns:
        feat["dev_x_volz"] = feat["dev_sma8"] * feat["volume_z_24"]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # keep feature count comfortably under limit
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
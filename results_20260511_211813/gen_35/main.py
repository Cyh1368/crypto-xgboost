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
    "max_depth": 5,
    "min_child_weight": 2,
    "reg_alpha": 0.0,
    "reg_lambda": 0.3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "learning_rate": 0.025,
    "max_bin": 256,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 80,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.15     # minimum acceptable pred_std / actual_std
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

    # Shorter, more reactive return stack
    r1 = close.pct_change(1)
    r2 = close.pct_change(2)
    r4 = close.pct_change(4)
    r8 = close.pct_change(8)
    r16 = close.pct_change(16)
    feat["ret_1"] = r1
    feat["ret_2"] = r2
    feat["ret_4"] = r4
    feat["ret_8"] = r8
    feat["ret_16"] = r16

    # Volatility and compression/expansion
    vol_4 = log_ret.rolling(4).std()
    vol_12 = log_ret.rolling(12).std()
    vol_48 = log_ret.rolling(48).std()
    feat["vol_4"] = vol_4
    feat["vol_12"] = vol_12
    feat["vol_48"] = vol_48
    feat["vol_ratio_4_48"] = vol_4 / (vol_48 + 1e-9)

    # RSI and mean reversion
    def rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    feat["rsi_7"] = rsi(close, 7)
    feat["rsi_14"] = rsi(close, 14)
    feat["rsi_slope"] = feat["rsi_7"].diff()

    # Trend / location
    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    feat["ema_gap_8_21"] = (ema_8 - ema_21) / (close + 1e-9)
    feat["ema_gap_21_55"] = (ema_21 - ema_55) / (close + 1e-9)
    feat["ema_slope_8"] = ema_8.pct_change(2)
    feat["ema_slope_55"] = ema_55.pct_change(4)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_z"] = (close - bb_mid) / (bb_std + 1e-9)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_norm"] = feat["atr_14"] / (close + 1e-9)

    # Microstructure candle shape
    candle_range = (high - low).replace(0, np.nan)
    feat["candle_body"] = (close - open_) / (candle_range + 1e-9)
    feat["upper_wick"] = (high - np.maximum(close, open_)) / (candle_range + 1e-9)
    feat["lower_wick"] = (np.minimum(close, open_) - low) / (candle_range + 1e-9)
    feat["close_loc"] = (close - low) / (candle_range + 1e-9)

    # Range / breakout
    hi_20 = high.rolling(20).max()
    lo_20 = low.rolling(20).min()
    feat["range_pos_20"] = (close - lo_20) / ((hi_20 - lo_20) + 1e-9)
    feat["range_width_20"] = (hi_20 - lo_20) / (close + 1e-9)

    # Volume / participation
    feat["volume_ratio_5"] = volume / (volume.rolling(5).mean() + 1e-9)
    feat["volume_ratio_20"] = volume / (volume.rolling(20).mean() + 1e-9)
    feat["volume_z_20"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    feat["volume_chg_1"] = volume.pct_change(1)
    feat["vwap_dev_20"] = (close - (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)) / (close + 1e-9)

    # Regime proxies
    realized_vol = log_ret.rolling(20).std()
    feat["vol_regime"] = realized_vol / (realized_vol.rolling(96).mean() + 1e-9)
    feat["trend_strength_24"] = (close / close.shift(24) - 1).abs() * 10000
    feat["efficiency_ratio_14"] = (close - close.shift(14)).abs() / (close.diff().abs().rolling(14).sum() + 1e-9)

    # Interactions that switch behavior by regime
    feat["vol_x_mom"] = feat["vol_regime"] * r4
    feat["vol_x_revert"] = feat["vol_regime"] * (close / close.rolling(20).mean() - 1)
    feat["trend_x_rsi"] = feat["trend_strength_24"] * (feat["rsi_14"] / 100.0 - 0.5)
    feat["range_x_volume"] = feat["range_width_20"] * feat["volume_ratio_5"]
    feat["wick_imbalance"] = feat["lower_wick"] - feat["upper_wick"]
    feat["momentum_spread"] = (r1 + r2) - (r8 + r16)

    # Time features
    idx = df.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    # Orderbook features, if present
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

    # Funding features, if present
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding_rate"] = fr
        feat["funding_ma_8h"] = fr.rolling(32).mean()
        feat["funding_chg_1"] = fr.diff(1)
        feat["funding_chg_4"] = fr.diff(4)
        feat["funding_z_32"] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)

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
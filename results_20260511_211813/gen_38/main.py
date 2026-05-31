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
    "n_estimators": 850,
    "max_depth": 5,
    "min_child_weight": 4,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.03,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 60,
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

    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    open_ = df['open'].astype(float)

    # --- Returns & Trend Stack ---
    r1 = close.pct_change(1)
    r3 = close.pct_change(3)
    r6 = close.pct_change(6)
    r12 = close.pct_change(12)

    for lag in [1, 2, 4, 8, 16, 32]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    feat['trend_stack'] = np.sign(r1) + np.sign(r3) + np.sign(r6) + np.sign(r12)
    feat['ret_accel_3'] = r1 - r3 / 3.0
    feat['ret_accel_6'] = r3 - r6 / 2.0

    # --- Volatility & Regime ---
    log_ret = np.log(close / close.shift(1))
    rv_8 = log_ret.rolling(8).std()
    rv_24 = log_ret.rolling(24).std()
    rv_96 = log_ret.rolling(96).std()

    feat['vol_8'] = rv_8
    feat['vol_ratio_8_96'] = rv_8 / (rv_96 + 1e-9)
    feat['vol_accel'] = rv_8 / (rv_24 + 1e-9) - 1.0
    feat['vol_state'] = np.tanh((feat['vol_ratio_8_96'] - 1.0) * 1.5)

    # --- Moving Average Gaps (Trend Distance) ---
    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    feat['ema_gap_8_21'] = (ema_8 - ema_21) / (close + 1e-9)
    feat['ema_gap_21_55'] = (ema_21 - ema_55) / (close + 1e-9)

    # --- RSI & Oscillators ---
    def get_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    feat['rsi_7'] = get_rsi(close, 7)
    feat['rsi_14'] = get_rsi(close, 14)
    feat['rsi_slope'] = feat['rsi_7'].diff()

    # --- Candle Geometry & Microstructure ---
    bar_range = (high - low).replace(0, np.nan)
    feat['candle_body'] = (close - open_) / (bar_range + 1e-9)
    feat['upper_wick'] = (high - np.maximum(close, open_)) / (bar_range + 1e-9)
    feat['lower_wick'] = (np.minimum(close, open_) - low) / (bar_range + 1e-9)
    feat['wick_imbalance'] = feat['lower_wick'] - feat['upper_wick']
    feat['close_loc'] = (close - low) / (bar_range + 1e-9)

    # --- Range & Breakout Pressure ---
    hi_20 = high.rolling(20).max()
    lo_20 = low.rolling(20).min()
    feat['range_pos_20'] = (close - lo_20) / (hi_20 - lo_20 + 1e-9)
    feat['range_width_20'] = (hi_20 - lo_20) / (close + 1e-9)

    # --- Volume & VWAP ---
    vol_mean_20 = volume.rolling(20).mean()
    feat['volume_ratio_20'] = volume / (vol_mean_20 + 1e-9)
    feat['volume_z_20'] = (volume - vol_mean_20) / (volume.rolling(20).std() + 1e-9)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev_20'] = (close - vwap_20) / (close.rolling(20).std() + 1e-9)

    # --- Rolling Quantile Positions (Scale Invariant) ---
    feat['ret_qpos_60'] = r1.rolling(60).rank(pct=True)
    feat['vol_qpos_60'] = volume.rolling(60).rank(pct=True)
    feat['close_qpos_60'] = close.rolling(60).rank(pct=True)

    # --- Time Features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook features ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'], dtype=float)
                asks = np.array(row['asks'], dtype=float)
                if bids.ndim != 2 or asks.ndim != 2: return 0.0
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except: return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

    # --- Funding features ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].astype(float)
        feat['funding_rate'] = fr
        feat['funding_ma_8h'] = fr.rolling(32).mean()
        feat['funding_chg_1'] = fr.diff(1)

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
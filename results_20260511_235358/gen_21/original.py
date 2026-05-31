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
    "n_estimators": 650,
    "max_depth": 3,
    "min_child_weight": 6,
    "reg_alpha": 0.2,
    "reg_lambda": 2.5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 60,
}

CLIP_PERCENTILE = 97          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
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

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    open_ = df['open']

    log_ret = np.log(close / close.shift(1))
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    vol_60 = log_ret.rolling(60).std()

    # --- Short-horizon returns / normalized returns ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        r = close.pct_change(lag)
        feat[f'ret_{lag}'] = r
        feat[f'ret_z_{lag}'] = (r - r.rolling(20).mean()) / (r.rolling(20).std() + 1e-9)

    # --- Regime-invariant volatility features ---
    feat['vol_5'] = vol_5
    feat['vol_10'] = vol_10
    feat['vol_20'] = vol_20
    feat['vol_z_20'] = (vol_20 - vol_20.rolling(60).mean()) / (vol_20.rolling(60).std() + 1e-9)
    feat['vol_rank_60'] = vol_20.rolling(60).rank(pct=True)
    feat['vol_trend'] = vol_20 / (vol_20.shift(10) + 1e-9) - 1

    # --- RSI / momentum oscillators on short windows ---
    for period in [5, 9, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Bollinger / position with shorter windows ---
    bb_mid_10 = close.rolling(10).mean()
    bb_std_10 = close.rolling(10).std()
    bb_mid_20 = close.rolling(20).mean()
    bb_std_20 = close.rolling(20).std()
    feat['bb_pct_10'] = (close - bb_mid_10) / (2 * bb_std_10 + 1e-9)
    feat['bb_pct_20'] = (close - bb_mid_20) / (2 * bb_std_20 + 1e-9)
    feat['close_pos_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-9)

    # --- ATR / candle structure ---
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_7'] = tr.rolling(7).mean()
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm_7'] = feat['atr_7'] / close
    feat['range_norm'] = (high - low) / close
    feat['body_ratio'] = (close - open_).abs() / ((high - low) + 1e-9)

    # --- MACD / trend efficiency ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd = ema8 - ema21
    feat['macd_signal'] = macd - macd.ewm(span=5, adjust=False).mean()
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)

    # --- Volume / flow ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    feat['volume_ratio_5'] = volume / (vol_ma_5 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma_20 + 1e-9)
    feat['volume_z_20'] = (volume - vol_ma_20) / (volume.rolling(20).std() + 1e-9)
    feat['vwap_dev'] = (close - (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)) / (close + 1e-9)

    # --- Character change features ---
    feat['vol_trend_fast'] = vol_10 / (vol_10.shift(5) + 1e-9) - 1
    feat['vol_trend_slow'] = vol_20 / (vol_20.shift(10) + 1e-9) - 1
    feat['eff_trend'] = feat['efficiency_10'] - feat['efficiency_10'].shift(5)
    feat['autocorr_5'] = log_ret.rolling(20).corr(log_ret.shift(1))
    feat['autocorr_trend'] = feat['autocorr_5'] - feat['autocorr_5'].rolling(10).mean()

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Interactions ---
    ret_3 = close.pct_change(3)
    feat['momentum_vol_interaction'] = ret_3 * vol_20
    feat['momentum_x_volregime'] = ret_3 * (vol_20 / (vol_20.rolling(60).mean() + 1e-9))
    feat['volume_price_pressure'] = feat['volume_ratio_20'] * feat['bb_pct_20']
    feat['eff_x_voltrend'] = feat['efficiency_10'] * feat['vol_trend_fast']

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
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_momentum'] = fr.diff(4)
        feat['funding_z_32'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)
        feat['funding_x_voltrend'] = fr * feat['vol_trend_fast']

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
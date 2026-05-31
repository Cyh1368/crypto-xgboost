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
    "n_estimators": 700,
    "max_depth": 4,
    "min_child_weight": 6,
    "gamma": 0.5,
    "reg_alpha": 0.25,
    "reg_lambda": 2.0,
    "subsample": 0.75,
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

    def rmean(s, w):
        return s.rolling(w).mean()

    def rstd(s, w):
        return s.rolling(w).std()

    def rz(s, w):
        return (s - rmean(s, w)) / (rstd(s, w) + 1e-9)

    # --- Short-horizon returns / momentum ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        feat[f'ret_{lag}'] = close.pct_change(lag)
        feat[f'ret_z_{lag}'] = rz(close.pct_change(1), max(5, min(20, lag * 2)))

    # --- Volatility and regime normalization ---
    for w in [5, 10, 20]:
        feat[f'vol_{w}'] = log_ret.rolling(w).std()
        feat[f'vol_z_{w}'] = rz(log_ret.rolling(w).std(), 60)

    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_regime'] = vol_10 / (vol_20.rolling(60).mean() + 1e-9)
    feat['vol_regime_z'] = rz(feat['vol_regime'], 60)
    feat['vol_trend_20'] = feat['vol_regime'] - feat['vol_regime'].shift(5)

    # --- Trend / efficiency / autocorrelation trends ---
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_trend'] = feat['efficiency_10'] - feat['efficiency_10'].shift(5)
    feat['autocorr_10'] = log_ret.rolling(10).corr(log_ret.shift(1))
    feat['autocorr_trend'] = feat['autocorr_10'] - feat['autocorr_10'].shift(5)

    # --- RSI / price position ---
    for period in [5, 10, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)
        feat[f'rsi_z_{period}'] = rz(feat[f'rsi_{period}'], 60)

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    feat['price_sma10_dev'] = (close - sma_10) / (sma_10 + 1e-9)
    feat['price_sma20_dev'] = (close - sma_20) / (sma_20 + 1e-9)

    # --- Bollinger / range ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['hl_range'] = (high - low) / (close + 1e-9)
    feat['hl_range_z'] = rz(feat['hl_range'], 60)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_10'] = tr.rolling(10).mean() / (close + 1e-9)
    feat['atr_20'] = tr.rolling(20).mean() / (close + 1e-9)

    # --- Volume / activity ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    feat['volume_z_20'] = rz(volume, 20)
    feat['volume_z_60'] = rz(volume, 60)
    feat['vol_price_pressure'] = feat['volume_ratio_5'] * feat['price_sma10_dev']

    vwap_10 = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + 1e-9)
    feat['vwap_dev_10'] = (close - vwap_10) / (close + 1e-9)

    # --- Time features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    # --- Interactions that often generalize better than raw terms ---
    feat['momentum_vol_interaction'] = close.pct_change(3) * feat['vol_regime']
    feat['momentum_vol_interaction_z'] = rz(feat['momentum_vol_interaction'], 60)
    feat['momentum_regime'] = close.pct_change(3) / (vol_20 + 1e-9)

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
        feat['funding_8h_ma'] = fr.rolling(32).mean()
        feat['funding_momentum'] = fr.diff(4)
        feat['funding_x_volregime'] = fr * feat['vol_regime']

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
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
    "n_estimators": 350,
    "max_depth": 3,
    "min_child_weight": 25,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 45,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build highly robust, regime-invariant features using rolling z-scores
    and market efficiency indicators.
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    open_ = df['open']
    log_ret = np.log(close / close.shift(1))

    # Helper for rolling z-scores
    def rolling_z(series, w_lookback, w_stats=50):
        m = series.rolling(w_stats).mean()
        s = series.rolling(w_stats).std()
        return (series - m) / (s + 1e-9)

    # --- Returns & Momentum (Local Z-Score) ---
    for w in [1, 3, 6, 12]:
        r = close.pct_change(w)
        feat[f'ret_z_{w}'] = rolling_z(r, w, 60)

    # --- Volatility Dynamics (Local Z-Score) ---
    for w in [5, 15, 30]:
        v = log_ret.rolling(w).std()
        feat[f'vol_z_{w}'] = rolling_z(v, w, 100)

    # --- Rate of character change ---
    vol_fast = log_ret.rolling(5).std()
    vol_slow = log_ret.rolling(20).std()
    feat['vol_regime_ratio'] = vol_fast / (vol_slow + 1e-9)
    feat['vol_trend'] = vol_fast.diff(5) / (vol_slow + 1e-9)

    # Kaufman's Efficiency Ratio (ER) - measure of noise vs trend
    def get_er(series, w):
        direction = (series - series.shift(w)).abs()
        volatility = series.diff().abs().rolling(w).sum()
        return direction / (volatility + 1e-9)

    feat['er_10'] = get_er(close, 10)
    feat['er_trend'] = feat['er_10'].diff(5)

    # --- Normalized Oscillators ---
    for w in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rsi = 100 - 100 / (1 + (gain / (loss + 1e-9)))
        feat[f'rsi_z_{w}'] = rolling_z(rsi, w, 60)

    # --- Price position and mean reversion ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['vwap_dev_z'] = rolling_z((close - (close * volume).rolling(15).sum() / (volume.rolling(15).sum() + 1e-9)), 15, 60)

    # --- Force Robust Interactions ---
    feat['mom_vol_adj'] = close.pct_change(5) / (vol_slow + 1e-9)
    feat['mom_er_adj'] = close.pct_change(5) * feat['er_10']

    # --- Volume Dynamics ---
    vol_ma = volume.rolling(20).mean()
    feat['vol_surge_z'] = rolling_z(volume / (vol_ma + 1e-9), 20, 60)
    feat['vol_ret_corr'] = log_ret.rolling(15).corr(volume.pct_change(1))

    # --- Candlestick Character ---
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_norm_z'] = rolling_z(tr.rolling(14).mean() / close, 14, 60)
    feat['candle_body_ratio'] = (close - open_).abs() / ((high - low) + 1e-9)
    feat['wick_balance'] = (high - df[['open', 'close']].max(axis=1)) / (df[['open', 'close']].min(axis=1) - low + 1e-9)
    feat['close_pos_range'] = (close - low.rolling(15).min()) / (high.rolling(15).max() - low.rolling(15).min() + 1e-9)

    # Autocorrelation (Persistence of trend)
    feat['autocorr_10'] = log_ret.rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

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
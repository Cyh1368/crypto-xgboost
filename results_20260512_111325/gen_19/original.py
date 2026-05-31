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
    "max_depth": 4,
    "min_child_weight": 4,
    "reg_alpha": 0.4,
    "reg_lambda": 2.0,
    "gamma": 0.15,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
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

    log_ret = np.log(close / close.shift(1))
    ret1 = close.pct_change(1)

    # --- Short-horizon returns / momentum ---
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        feat[f'ret_{lag}'] = close.pct_change(lag)
    feat['ret_1_z_20'] = (ret1 - ret1.rolling(20).mean()) / (ret1.rolling(20).std() + 1e-9)
    feat['ret_1_z_60'] = (ret1 - ret1.rolling(60).mean()) / (ret1.rolling(60).std() + 1e-9)

    # --- Shorter volatility / volatility trend ---
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_5'] = vol_5
    feat['vol_10'] = vol_10
    feat['vol_20'] = vol_20
    feat['vol_trend_10_20'] = (vol_10 - vol_20) / (vol_20 + 1e-9)
    feat['vol_trend_5_20'] = (vol_5 - vol_20) / (vol_20 + 1e-9)

    # --- RSI on shorter windows ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    for period in [5, 9, 14]:
        rs = gain.rolling(period).mean() / (loss.rolling(period).mean() + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Z-scored trend / position features ---
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_40 = close.rolling(40).mean()
    feat['price_sma10_dev'] = (close - sma_10) / (sma_10 + 1e-9)
    feat['price_sma20_dev_z'] = ((close - sma_20) / (sma_20 + 1e-9) - ((close - sma_20) / (sma_20 + 1e-9)).rolling(60).mean()) / (((close - sma_20) / (sma_20 + 1e-9)).rolling(60).std() + 1e-9)
    feat['sma_cross_10_40'] = (sma_10 - sma_40) / (sma_40 + 1e-9)

    # --- Range / candle structure ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_10'] = tr.rolling(10).mean()
    feat['atr_norm'] = feat['atr_10'] / (close + 1e-9)
    feat['hl_range'] = (high - low) / (close + 1e-9)
    feat['body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)

    # --- Efficiency / autocorrelation proxies ---
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat['efficiency_trend'] = feat['efficiency_10'] - feat['efficiency_20']
    feat['ret_autocorr_10'] = ret1.rolling(10).corr(ret1.shift(1))
    feat['ret_autocorr_20'] = ret1.rolling(20).corr(ret1.shift(1))
    feat['autocorr_trend'] = feat['ret_autocorr_10'] - feat['ret_autocorr_20']

    # --- Volume normalization / local regime ---
    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat['volume_z_20'] = (volume - vol_ma_20) / (vol_std_20 + 1e-9)
    feat['volume_ratio_10'] = volume / (vol_ma_10 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma_20 + 1e-9)

    # --- Regime indicators ---
    realized_vol_20 = vol_20
    realized_vol_60 = log_ret.rolling(60).std()
    feat['vol_regime'] = realized_vol_20 / (realized_vol_20.rolling(120).mean() + 1e-9)
    feat['vol_regime_fast'] = realized_vol_20 / (realized_vol_60 + 1e-9)
    feat['trend_strength_10'] = (close / close.shift(10) - 1).abs()
    feat['trend_strength_20'] = (close / close.shift(20) - 1).abs()

    # --- Interactions ---
    feat['momentum_vol_interaction'] = feat['ret_3'] * feat['vol_regime']
    feat['momentum_regime_fast'] = feat['ret_3'] * feat['vol_regime_fast']
    feat['volume_price_pressure'] = feat['volume_ratio_10'] * feat['price_sma10_dev']
    feat['efficiency_vol_interaction'] = feat['efficiency_trend'] * feat['vol_regime']

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

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
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
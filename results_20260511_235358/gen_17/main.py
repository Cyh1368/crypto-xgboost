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
    "min_child_weight": 4,
    "reg_alpha": 0.05,
    "reg_lambda": 0.8,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "learning_rate": 0.10,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.16     # minimum acceptable pred_std / actual_std
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

    log_close = np.log(close)
    log_ret = log_close.diff()

    # --- Short-horizon returns and rolling z-scores ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        feat[f'ret_{lag}'] = close.pct_change(lag)
    for w in [5, 10, 20]:
        rmu = log_ret.rolling(w).mean()
        rsd = log_ret.rolling(w).std()
        feat[f'ret_z_{w}'] = (log_ret - rmu) / (rsd + 1e-9)

    # --- Volatility and vol trend ---
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_5'] = vol_5
    feat['vol_10'] = vol_10
    feat['vol_20'] = vol_20
    feat['vol_trend_10_20'] = (vol_10 - vol_20) / (vol_20 + 1e-9)
    feat['vol_trend_5_10'] = (vol_5 - vol_10) / (vol_10 + 1e-9)
    feat['vol_ratio_5_20'] = vol_5 / (vol_20 + 1e-9)

    # --- RSI / momentum speed ---
    for period in [5, 9, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)
    feat['mom_3'] = close.pct_change(3)
    feat['mom_6'] = close.pct_change(6)
    feat['mom_accel'] = feat['mom_3'] - feat['mom_6']

    # --- Price location / mean reversion ---
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    feat['price_sma5_dev'] = (close - sma_5) / (sma_5 + 1e-9)
    feat['price_sma10_dev'] = (close - sma_10) / (sma_10 + 1e-9)
    feat['price_sma20_dev'] = (close - sma_20) / (sma_20 + 1e-9)
    feat['sma_cross_5_20'] = (sma_5 - sma_20) / (sma_20 + 1e-9)

    # --- Range / candle structure ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_10'] = tr.rolling(10).mean()
    feat['atr_norm_10'] = feat['atr_10'] / close
    feat['hl_range'] = (high - low) / close
    feat['candle_body_ratio'] = (close - open_).abs() / ((high - low) + 1e-9)

    # --- Volume / participation ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat['volume_ratio_5'] = volume / (vol_ma_5 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma_20 + 1e-9)
    feat['volume_z_20'] = (volume - vol_ma_20) / (vol_std_20 + 1e-9)
    feat['vol_price_interaction'] = feat['volume_ratio_20'] * feat['price_sma10_dev']

    # --- Regime features ---
    realized_vol_20 = vol_20
    realized_vol_40 = log_ret.rolling(40).std()
    feat['vol_regime'] = realized_vol_20 / (realized_vol_20.rolling(100).mean() + 1e-9)
    feat['vol_regime_fast'] = realized_vol_20 / (realized_vol_40 + 1e-9)
    feat['vol_rank_100'] = realized_vol_20.rolling(100).rank(pct=True)

    # --- Regime-change detection features ---
    # Volatility of volatility (vol_accel): measures if volatility is increasing/decreasing
    vol_accel_5 = vol_5.diff(5)
    vol_accel_10 = vol_10.diff(5)
    feat['vol_accel_5'] = vol_accel_5
    feat['vol_accel_10'] = vol_accel_10
    feat['vol_accel_ratio'] = vol_accel_5 / (vol_accel_10.abs() + 1e-9)

    # Momentum of momentum: second derivative of price
    feat['mom_accel_2'] = feat['mom_3'].diff(3)
    feat['mom_accel_z'] = (feat['mom_accel_2'] - feat['mom_accel_2'].rolling(15).mean()) / (feat['mom_accel_2'].rolling(15).std() + 1e-9)

    # Autocorrelation trend: is price mean-reverting or trending?
    feat['autocorr_10'] = log_ret.rolling(10).corr(log_ret.shift(1))
    feat['autocorr_trend'] = feat['autocorr_10'].diff(5)

    # Regime-invariant transformations with shorter windows
    v_ma_50 = vol_10.rolling(50).mean()
    v_sd_50 = vol_10.rolling(50).std()
    feat['vol_z_50'] = (vol_10 - v_ma_50) / (v_sd_50 + 1e-9)
    feat['mom_z_10'] = (close / close.shift(10) - 1) / (vol_10 * np.sqrt(10) + 1e-9)
    feat['price_z_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)

    trend_10 = close.pct_change(10)
    trend_20 = close.pct_change(20)
    feat['trend_dir_10'] = np.sign(trend_10)
    feat['trend_dir_20'] = np.sign(trend_20)
    feat['trend_strength_10'] = trend_10.abs() * 10000
    feat['trend_strength_20'] = trend_20.abs() * 10000

    # --- Interaction: momentum × regime and regime-change signals ---
    feat['momentum_vol_interaction'] = feat['mom_3'] * feat['vol_regime']
    feat['momentum_regime_fast'] = feat['mom_3'] * feat['vol_regime_fast']
    feat['mom_eff_interaction'] = feat['mom_3'] * feat['efficiency_10']
    feat['vol_accel_momentum'] = feat['vol_accel_5'] * feat['mom_3']
    feat['autocorr_momentum'] = feat['autocorr_10'] * feat['mom_3']

    # --- Efficiency / autocorrelation proxy ---
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat['efficiency_trend'] = feat['efficiency_10'] - feat['efficiency_20']
    feat['ret_accel_3'] = log_ret.diff(3)
    feat['ret_accel_6'] = log_ret.diff(6)
    feat['vol_ret_corr_10'] = log_ret.rolling(10).corr(volume.pct_change(1))
    feat['vol_ret_corr_20'] = log_ret.rolling(20).corr(volume.pct_change(1))

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
        feat['funding_8h_ma'] = fr.rolling(32).mean()   # 32 × 15min = 8h
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
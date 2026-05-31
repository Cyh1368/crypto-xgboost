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
    "n_estimators": 1400,
    "max_depth": 8,
    "min_child_weight": 1,
    "reg_alpha": 0.0005,
    "reg_lambda": 0.0015,
    "subsample": 0.88,
    "colsample_bytree": 0.9,
    "learning_rate": 0.045,
    "gamma": 0.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 80,
}

CLIP_PERCENTILE = 97          # keep more directional signal in the target
MIN_PRED_STD_RATIO = 0.18     # encourage stronger prediction dispersion
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
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Use past-only shifted anchors for all engineered features
    c1 = close.shift(1)
    o1 = open_.shift(1)
    h1 = high.shift(1)
    l1 = low.shift(1)
    v1 = volume.shift(1)

    log_ret = np.log(c1 / c1.shift(1))
    ret1 = c1.pct_change(1)

    # --- Multi-horizon momentum ---
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        feat[f'ret_{lag}'] = c1.pct_change(lag)

    feat['ret_2_1_accel'] = feat['ret_{0}'.format(2)] - feat['ret_{0}'.format(1)]
    feat['ret_6_3_accel'] = feat['ret_{0}'.format(6)] - feat['ret_{0}'.format(3)]
    feat['ret_12_6_accel'] = feat['ret_{0}'.format(12)] - feat['ret_{0}'.format(6)]

    # --- Volatility / compression-expansion ---
    for w in [5, 10, 20, 60]:
        feat[f'vol_{w}'] = log_ret.rolling(w).std()

    feat['vol_ratio_5_20'] = feat['vol_5'] / (feat['vol_20'] + 1e-9)
    feat['vol_ratio_20_60'] = feat['vol_20'] / (feat['vol_60'] + 1e-9)

    tr = pd.concat([
        h1 - l1,
        (h1 - c1.shift(1)).abs(),
        (l1 - c1.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm'] = feat['atr_14'] / (c1 + 1e-9)
    feat['range_pct'] = (h1 - l1) / (c1 + 1e-9)

    # Volatility regime and compression flags
    vol_ref = feat['vol_20'].rolling(120).mean()
    feat['vol_regime'] = feat['vol_20'] / (vol_ref + 1e-9)
    feat['vol_high'] = (feat['vol_regime'] > 1.10).astype(float)
    feat['vol_low'] = (feat['vol_regime'] < 0.90).astype(float)
    feat['vol_surge'] = (feat['vol_20'] > feat['vol_20'].rolling(30).mean() + feat['vol_20'].rolling(30).std()).astype(float)
    feat['vol_compress'] = (feat['vol_20'] < feat['vol_20'].rolling(30).mean() - 0.5 * feat['vol_20'].rolling(30).std()).astype(float)

    # --- Trend / mean reversion ---
    sma_10 = c1.rolling(10).mean()
    sma_20 = c1.rolling(20).mean()
    sma_60 = c1.rolling(60).mean()
    ema_12 = c1.ewm(span=12, adjust=False).mean()
    ema_26 = c1.ewm(span=26, adjust=False).mean()
    ema_50 = c1.ewm(span=50, adjust=False).mean()

    feat['dist_sma20'] = (c1 - sma_20) / (sma_20 + 1e-9)
    feat['dist_sma60'] = (c1 - sma_60) / (sma_60 + 1e-9)
    feat['sma_slope_20'] = sma_20.pct_change(5)
    feat['sma_slope_60'] = sma_60.pct_change(10)
    feat['ema_spread_12_26'] = (ema_12 - ema_26) / (c1 + 1e-9)
    feat['ema_spread_12_50'] = (ema_12 - ema_50) / (c1 + 1e-9)
    feat['trend_strength_60'] = (c1 / c1.shift(60) - 1.0) / (feat['vol_20'] + 1e-9)

    feat['trend_up'] = ((c1 > sma_20) & (sma_20 > sma_60)).astype(float)
    feat['trend_down'] = ((c1 < sma_20) & (sma_20 < sma_60)).astype(float)
    feat['trend_regime'] = feat['trend_up'] - feat['trend_down']

    # --- Oscillators ---
    def rsi(series, period):
        d = series.diff()
        gain = d.clip(lower=0).rolling(period).mean()
        loss = (-d.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100.0 - 100.0 / (1.0 + rs)

    feat['rsi_6'] = rsi(c1, 6)
    feat['rsi_14'] = rsi(c1, 14)
    feat['rsi_28'] = rsi(c1, 28)
    feat['rsi_6_14_diff'] = feat['rsi_6'] - feat['rsi_14']
    feat['rsi_14_28_diff'] = feat['rsi_14'] - feat['rsi_28']
    feat['rsi_overbought'] = (feat['rsi_14'] > 70).astype(float)
    feat['rsi_oversold'] = (feat['rsi_14'] < 30).astype(float)

    bb_mid = c1.rolling(20).mean()
    bb_std = c1.rolling(20).std()
    feat['bb_z'] = (c1 - bb_mid) / (bb_std + 1e-9)
    feat['bb_width'] = (4 * bb_std) / (bb_mid + 1e-9)

    # --- Candle anatomy ---
    feat['body_pct'] = (c1 - o1) / (h1 - l1 + 1e-9)
    feat['upper_wick_pct'] = (h1 - np.maximum(c1, o1)) / (h1 - l1 + 1e-9)
    feat['lower_wick_pct'] = (np.minimum(c1, o1) - l1) / (h1 - l1 + 1e-9)
    feat['close_pos_range'] = (c1 - l1) / (h1 - l1 + 1e-9)
    feat['gap_pct'] = (o1 - c1.shift(1)) / (c1.shift(1) + 1e-9)
    feat['body_to_range'] = (c1 - o1).abs() / (h1 - l1 + 1e-9)

    # --- Volume / flow ---
    vma_5 = v1.rolling(5).mean()
    vma_20 = v1.rolling(20).mean()
    vstd_20 = v1.rolling(20).std()

    feat['volume_ratio_5'] = v1 / (vma_5 + 1e-9)
    feat['volume_ratio_20'] = v1 / (vma_20 + 1e-9)
    feat['volume_z_20'] = (v1 - vma_20) / (vstd_20 + 1e-9)
    feat['log_volume'] = np.log1p(v1)
    feat['vol_change_5'] = v1.pct_change(5)
    feat['vol_change_20'] = v1.pct_change(20)
    feat['vol_x_ret1'] = feat['volume_ratio_5'] * ret1

    vwap_20 = (c1 * v1).rolling(20).sum() / (v1.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (c1 - vwap_20) / (c1 + 1e-9)
    feat['vpt_10'] = (ret1 * v1).rolling(10).sum() / (vma_20 + 1e-9)

    # --- Time / seasonality ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime interactions ---
    feat['ret1_x_volreg'] = feat['ret_1'] * feat['vol_regime']
    feat['ret6_x_volreg'] = feat['ret_6'] * feat['vol_regime']
    feat['ret12_x_volreg'] = feat['ret_12'] * feat['vol_regime']
    feat['trend_x_volreg'] = feat['trend_strength_60'] * feat['vol_regime']
    feat['rsi_x_volreg'] = feat['rsi_14'] * feat['vol_regime']
    feat['dist20_x_trend'] = feat['dist_sma20'] * feat['trend_regime']
    feat['mom_trend_align'] = np.sign(feat['ret_6']) * feat['trend_regime']
    feat['mom_trend_diverge'] = -np.sign(feat['ret_6']) * feat['trend_regime']
    feat['mom_vs_rsi'] = np.sign(feat['ret_6']) * np.sign(50.0 - feat['rsi_14'])

    # --- Past-only adaptive / normalized momentum ---
    recent_vol = log_ret.rolling(10).std()
    feat['adaptive_ret6'] = feat['ret_6'] / (recent_vol + 1e-9)
    feat['adaptive_ret12'] = feat['ret_12'] / (recent_vol + 1e-9)
    feat['momentum_shock'] = feat['ret_6'].abs() / (feat['ret_6'].rolling(20).std() + 1e-9)
    feat['ret_accel_scaled'] = feat['ret_12_6_accel'] / (feat['vol_20'] + 1e-9)

    # --- High-contrast splits for trees ---
    feat['mom_positive'] = (feat['ret_6'] > 0).astype(float)
    feat['mom_strong'] = (feat['ret_6'].abs() > feat['ret_6'].rolling(20).std()).astype(float)
    feat['strong_mom_low_vol'] = ((feat['ret_6'].abs() > feat['ret_6'].rolling(20).std()) & (feat['vol_low'] > 0)).astype(float)
    feat['weak_mom_high_vol'] = ((feat['ret_6'].abs() < feat['ret_6'].rolling(20).std()) & (feat['vol_high'] > 0)).astype(float)

    # --- Orderbook (if available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi_tau(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                if bids.ndim != 2 or asks.ndim != 2:
                    return 0.0
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0

        ob = df.shift(1)
        feat['obi_tau1'] = ob.apply(lambda r: obi_tau(r, 1), axis=1)
        feat['obi_tau3'] = ob.apply(lambda r: obi_tau(r, 3), axis=1)
        feat['obi_diff'] = feat['obi_tau1'] - feat['obi_tau3']

    # --- Funding rate (if available) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].shift(1)
        feat['funding_rate'] = fr
        feat['funding_ma_8h'] = fr.rolling(32).mean()
        feat['funding_z_32'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)
        feat['funding_change_8'] = fr.diff(8)
        feat['funding_extreme'] = (fr.abs() > fr.rolling(64).std()).astype(float)

    # Cleanup
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.ffill().fillna(0.0)

    # Keep feature count under limit
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
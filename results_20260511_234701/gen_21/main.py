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
    "max_depth": 5,
    "min_child_weight": 5,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 97          # slightly less aggressive clipping to preserve slope
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-adaptive features using short rolling windows and local normalization.
    All features use only past data.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    open_ = df['open']
    log_ret = np.log(close / close.shift(1))

    def z_score(s, window=20):
        m = s.rolling(window).mean()
        sd = s.rolling(window).std()
        return (s - m) / (sd + 1e-9)

    # short returns + local normalization
    for lag in [1, 2, 3, 6, 12]:
        r = close.pct_change(lag)
        feat[f'ret_{lag}'] = r
        feat[f'ret_z_{lag}'] = z_score(r, 20)

    # volatility levels and trends
    vol_5 = log_ret.rolling(5).std()
    vol_10 = log_ret.rolling(10).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_5'] = vol_5
    feat['vol_10'] = vol_10
    feat['vol_20'] = vol_20
    feat['vol_z_20'] = z_score(vol_20, 30)
    feat['vol_trend_5_20'] = (vol_5 - vol_20) / (vol_20 + 1e-9)
    feat['vol_accel'] = vol_5.diff(3)

    # regime proxies
    feat['vol_regime'] = vol_10 / (vol_20 + 1e-9)
    feat['vol_regime_z'] = z_score(feat['vol_regime'], 30)

    # RSI / momentum with shorter memory
    for period in [6, 9, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # Bollinger / position features
    bb_mid_10 = close.rolling(10).mean()
    bb_std_10 = close.rolling(10).std()
    bb_mid_20 = close.rolling(20).mean()
    bb_std_20 = close.rolling(20).std()
    feat['bb_pct_10'] = (close - bb_mid_10) / (2 * bb_std_10 + 1e-9)
    feat['bb_pct_20'] = (close - bb_mid_20) / (2 * bb_std_20 + 1e-9)

    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['close_pos_10'] = (close - low_10) / (high_10 - low_10 + 1e-9)
    feat['close_pos_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # ATR / range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_10'] = tr.rolling(10).mean() / (close + 1e-9)
    feat['hl_range_10'] = ((high - low) / close).rolling(10).mean()

    # trend / efficiency
    feat['efficiency_5'] = (close - close.shift(5)).abs() / (close.diff().abs().rolling(5).sum() + 1e-9)
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_trend'] = (feat['efficiency_5'] - feat['efficiency_10']) / (feat['efficiency_10'] + 1e-9)
    feat['trend_dir_10'] = np.sign(close / close.shift(10) - 1)
    feat['trend_strength_10'] = (close / close.shift(10) - 1).abs()

    # normalized MACD / momentum interaction
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    macd = ema8 - ema17
    feat['macd_z'] = z_score(macd, 20)
    ret_3 = close.pct_change(3)
    feat['mom_vol_adj'] = ret_3 / (vol_10 + 1e-9)
    feat['momentum_x_volregime'] = ret_3 * feat['vol_regime']

    # volume and pressure
    rel_vol_5 = volume / (volume.rolling(5).mean() + 1e-9)
    rel_vol_10 = volume / (volume.rolling(10).mean() + 1e-9)
    feat['rel_vol_5'] = rel_vol_5
    feat['rel_vol_10'] = rel_vol_10
    feat['rel_vol_z'] = z_score(rel_vol_10, 20)
    feat['volume_price_pressure'] = rel_vol_10 * ((close - bb_mid_10) / (bb_mid_10 + 1e-9))

    # candle structure
    feat['candle_body_ratio'] = (close - open_).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - low) / ((high - low) + 1e-9)

    # time features
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # orderbook & funding (conditional)
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + 1e-9)
            except:
                return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)

    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_mom'] = fr.diff(2)
        feat['funding_z'] = z_score(fr, 20)
        feat['funding_x_volregime'] = fr * feat['vol_regime']

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
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
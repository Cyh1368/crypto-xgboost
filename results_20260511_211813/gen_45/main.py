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
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 80,
}

CLIP_PERCENTILE = 96          # clip targets at this percentile (both tails)
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

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    # --- Returns ---
    for lag in [1, 2, 4, 8, 16]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    # --- Volatility ---
    log_ret = np.log(close / close.shift(1))
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    vol_60 = log_ret.rolling(60).std()
    feat['vol_5'] = vol_5
    feat['vol_20'] = vol_20
    feat['vol_60'] = vol_60
    feat['vol_ratio_5_20'] = vol_5 / (vol_20 + 1e-9)
    feat['vol_ratio_20_60'] = vol_20 / (vol_60 + 1e-9)

    # --- RSI ---
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm'] = feat['atr_14'] / close

    # --- Volume ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features ---
    realized_vol = log_ret.rolling(20).std()
    feat['vol_regime'] = realized_vol / (realized_vol.rolling(200).mean() + 1e-9)

    # --- Momentum and trend ---
    ret_1 = close.pct_change(1)
    ret_3 = close.pct_change(3)
    ret_6 = close.pct_change(6)
    ret_12 = close.pct_change(12)

    feat['momentum_decay_6'] = (ret_1 * 0.5 + ret_3 * 0.3 + ret_6 * 0.2)
    feat['momentum_fast'] = (ret_1 * 0.7 + ret_3 * 0.3)
    feat['momentum_slow'] = (ret_6 * 0.6 + ret_12 * 0.4)

    # Return acceleration
    feat['ret_accel_3'] = ret_1 - ret_3 / 3.0
    feat['ret_accel_6'] = ret_3 - ret_6 / 2.0

    # Volatility-adjusted returns
    feat['vol_weighted_ret'] = ret_1 / (vol_5 + 1e-6)
    feat['vol_adjusted_momentum'] = feat['momentum_decay_6'] / (vol_5 + 1e-6)

    # Rolling return z-score
    ret_mean_20 = log_ret.rolling(20).mean()
    ret_std_20 = log_ret.rolling(20).std()
    feat['ret_z_20'] = (log_ret - ret_mean_20) / (ret_std_20 + 1e-9)

    # Volume z-score
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat['volume_z_20'] = (volume - vol_mean_20) / (vol_std_20 + 1e-9)

    # Candle microstructure (high-signal features)
    candle_range = (high - low).replace(0, np.nan)
    feat['candle_body'] = (close - df['open']) / (candle_range + 1e-9)
    feat['upper_wick'] = (high - np.maximum(close, df['open'])) / (candle_range + 1e-9)
    feat['lower_wick'] = (np.minimum(close, df['open']) - low) / (candle_range + 1e-9)
    feat['wick_imbalance'] = feat['lower_wick'] - feat['upper_wick']

    # Price location within recent range
    roll_high_20 = high.rolling(20).max()
    roll_low_20 = low.rolling(20).min()
    feat['range_pos_20'] = (close - roll_low_20) / ((roll_high_20 - roll_low_20) + 1e-9)
    feat['range_width_20'] = (roll_high_20 - roll_low_20) / (close + 1e-9)

    # Direction persistence (sign consistency)
    sign_ret_1 = np.sign(ret_1)
    feat['dir_persist_8'] = sign_ret_1.rolling(8).mean()
    feat['dir_persist_24'] = sign_ret_1.rolling(24).mean()

    # Return quantile position (scale-invariant)
    feat['ret_qpos_60'] = ret_1.rolling(60).rank(pct=True)
    feat['vol_qpos_60'] = volume.rolling(60).rank(pct=True)

    # Simple, direct interactions
    feat['vol_state'] = np.tanh((feat['vol_ratio_5_20'] - 1.0) * 1.5)
    feat['vol_regime_x_momentum'] = feat['vol_regime'] * feat['momentum_decay_6']
    feat['vol_state_x_momentum'] = feat['vol_state'] * feat['momentum_decay_6']
    feat['range_x_volume'] = feat['range_pos_20'] * (feat['volume_ratio_5'] - 1.0)

    # RSI slope
    feat['rsi_7_slope'] = feat['rsi_7'].diff()

    # --- Orderbook (if real data available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def get_obi_series(bids_col, asks_col, tau=1):
            obi_list = []
            weights = np.exp(-tau * np.arange(10))
            for b, a in zip(bids_col, asks_col):
                try:
                    bv = np.sum(np.array(b)[:10, 1] * weights[:len(b)])
                    av = np.sum(np.array(a)[:10, 1] * weights[:len(a)])
                    obi_list.append((bv - av) / (bv + av + 1e-9))
                except: obi_list.append(0.0)
            return obi_list
        feat['obi_tau1'] = get_obi_series(df['bids'], df['asks'], 1)

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_8h_ma'] = fr.rolling(32).mean()
        feat['funding_chg_4'] = fr.diff(4)

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
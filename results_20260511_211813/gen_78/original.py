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
    "n_estimators": 1000,
    "max_depth": 6,
    "min_child_weight": 5,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.02,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 80,
}

CLIP_PERCENTILE = 95          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.15     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features must use only past data (no lookahead).
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    open_ = df['open'].astype(float)

    # --- Returns and Trend Stack ---
    r1 = close.pct_change(1)
    r3 = close.pct_change(3)
    r6 = close.pct_change(6)
    r12 = close.pct_change(12)
    feat['ret_1'], feat['ret_3'], feat['ret_6'], feat['ret_12'] = r1, r3, r6, r12
    feat['trend_stack'] = np.sign(r1) + np.sign(r3) + np.sign(r6) + np.sign(r12)
    feat['ret_accel_3'] = r1 - r3 / 3.0

    # --- Volatility and Vol State ---
    log_ret = np.log(close / close.shift(1))
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    vol_60 = log_ret.rolling(60).std()
    feat['vol_5'], feat['vol_20'], feat['vol_60'] = vol_5, vol_20, vol_60
    feat['vol_state'] = np.tanh((vol_5 / (vol_60 + 1e-9)) - 1.0)
    feat['vol_regime'] = vol_20 / (vol_20.rolling(200).mean() + 1e-9)

    # --- Oscillators ---
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)
    feat['rsi_6_slope'] = feat['rsi_6'].diff()
    feat['rsi_z_30'] = (feat['rsi_14'] - feat['rsi_14'].rolling(30).mean()) / (feat['rsi_14'].rolling(30).std() + 1e-9)

    # --- Mean Reversion / Gaps ---
    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    feat['ema_gap_fast'] = (ema_8 - ema_21) / (close + 1e-9)
    feat['ema_gap_slow'] = (ema_21 - ema_55) / (close + 1e-9)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)
    feat['streak_5'] = (r1 > 0).astype(int).rolling(5).sum() - (r1 < 0).astype(int).rolling(5).sum()

    # --- Volume Dynamics ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_z_20'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    feat['volume_chg_4'] = volume.pct_change(4)

    # --- Candle Microstructure ---
    bar_range = (high - low).replace(0, np.nan)
    feat['candle_body'] = (close - open_) / (bar_range + 1e-9)
    feat['upper_wick'] = (high - np.maximum(close, open_)) / (bar_range + 1e-9)
    feat['lower_wick'] = (np.minimum(close, open_) - low) / (bar_range + 1e-9)
    feat['close_loc'] = (close - low) / (bar_range + 1e-9)

    # --- Range and Breakout ---
    hi_20, lo_20 = high.rolling(20).max(), low.rolling(20).min()
    feat['range_pos_20'] = (close - lo_20) / ((hi_20 - lo_20) + 1e-9)
    feat['range_width_20'] = (hi_20 - lo_20) / (close + 1e-9)

    # --- Rolling Quantiles (Rank) ---
    feat['ret_qpos_60'] = r1.rolling(60).rank(pct=True)
    feat['vol_qpos_60'] = volume.rolling(60).rank(pct=True)

    # --- Interactions ---
    feat['momentum_decay_6'] = (r1 * 0.5 + r3 * 0.3 + r6 * 0.2)
    feat['vol_regime_x_mom'] = feat['vol_regime'] * feat['momentum_decay_6']
    feat['vol_mom_surge'] = feat['vol_state'] * feat['volume_ratio_5']

    # --- Time features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook / Funding ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def get_obi(b, a, tau=1):
            try:
                w = np.exp(-tau * np.arange(10))
                bv = np.sum(np.array(b)[:10, 1] * w[:len(b)])
                av = np.sum(np.array(a)[:10, 1] * w[:len(a)])
                return (bv - av) / (bv + av + 1e-9)
            except: return 0.0
        feat['obi_tau1'] = [get_obi(b, a, 1) for b, a in zip(df['bids'], df['asks'])]

    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].astype(float)
        feat['funding_rate'] = fr
        feat['funding_chg_4'] = fr.diff(4)

    return feat.replace([np.inf, -np.inf], np.nan).dropna()
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
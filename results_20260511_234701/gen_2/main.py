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
    "n_estimators": 500,
    "max_depth": 4,
    "min_child_weight": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-invariant features using rolling Z-scores and normalized indicators.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    log_ret = np.log(close / close.shift(1))

    def z_score(s, window=100):
        return (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-9)

    # --- Returns (Z-scored for regime invariance) ---
    for lag in [1, 4, 12]:
        feat[f'ret_z_{lag}'] = z_score(close.pct_change(lag), 100)

    # --- Volatility (Z-scored) ---
    vol_20 = log_ret.rolling(20).std()
    feat['vol_z_20'] = z_score(vol_20, 100)
    feat['vol_regime'] = vol_20 / (vol_20.rolling(100).mean() + 1e-9)

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat['rsi_14'] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands & ATR ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_norm'] = tr.rolling(14).mean() / (close + 1e-9)

    # --- Normalized MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feat['macd_norm'] = macd / (close.rolling(20).std() + 1e-9)

    # --- Volume (Z-scored) ---
    rel_vol = volume / (volume.rolling(20).mean() + 1e-9)
    feat['rel_vol_z'] = z_score(rel_vol, 100)

    # --- Candle Structure ---
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Range Position ---
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['close_pos_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    # --- Efficiency & Momentum ---
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat['mom_vol_adj'] = feat['ret_z_4'] / (vol_20 + 1e-9)

    # --- Time Features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook & Funding (Conditional) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + 1e-9)
            except: return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)

    if 'funding_rate' in df.columns:
        feat['funding_rate'] = df['funding_rate']
        feat['funding_mom'] = df['funding_rate'].diff(4)

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
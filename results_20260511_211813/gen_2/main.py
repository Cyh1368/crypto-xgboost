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
    "n_estimators": 600,
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_alpha": 0.05,
    "reg_lambda": 1.2,
    "subsample": 0.75,
    "colsample_bytree": 0.8,
    "learning_rate": 0.04,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # Slightly wider to allow more variance in predictions
MIN_PRED_STD_RATIO = 0.15     # Target ratio pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    Ensures 'shift by 1' constraint to prevent lookahead.
    """
    # Shift raw data by 1 to ensure that features for bar 't' use only 't-1' and earlier.
    d = df.shift(1)
    
    feat = pd.DataFrame(index=df.index)
    close = d['close']
    volume = d['volume']
    high = d['high']
    low = d['low']
    
    # Try to get 'open', else approximate using previous close
    if 'open' in d.columns:
        open_p = d['open']
    else:
        open_p = close.shift(1)

    # --- Returns (Momentum) ---
    for lag in [1, 2, 4, 12, 24, 96]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    # --- Volatility (Z-scored) ---
    log_ret = np.log(close / close.shift(1))
    for w in [10, 40]:
        rv = log_ret.rolling(w).std()
        feat[f'vol_{w}'] = rv
        feat[f'vol_z_{w}'] = (rv - rv.rolling(100).mean()) / (rv.rolling(100).std() + 1e-9)

    # --- RSI and RSI Momentum ---
    for period in [14, 28]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)
    feat['rsi_diff_6'] = feat['rsi_14'].diff(6)

    # --- Moving Average Deviations (Mean Reversion) ---
    for w in [20, 100]:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        feat[f'dist_ma_{w}'] = (close - ma) / (std + 1e-9)

    # --- Candle Shape Dynamics ---
    body = (close - open_p).abs()
    rng = (high - low).replace(0, 1e-9)
    feat['candle_body_ratio'] = (close - open_p) / rng
    feat['upper_wick_ratio'] = (high - close.combine(open_p, max)) / rng
    feat['lower_wick_ratio'] = (close.combine(open_p, min) - low) / rng

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feat['macd_val'] = macd / (close + 1e-9)
    feat['macd_sig'] = macd - macd.ewm(span=9, adjust=False).mean()

    # --- Volume Dynamics ---
    feat['volume_z_score'] = (volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 1e-9)
    feat['volume_ratio_long'] = volume / (volume.rolling(100).mean() + 1e-9)

    # --- Time periodicity ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Recent Range Position ---
    feat['dist_high_24h'] = (close / close.rolling(96).max()) - 1
    feat['dist_low_24h'] = (close / close.rolling(96).min()) - 1

    # --- Orderbook Imbalance ---
    if 'bids' in d.columns and 'asks' in d.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                if len(bids) == 0 or len(asks) == 0: return 0.0
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + 1e-9)
            except:
                return 0.0
        feat['obi_tau1'] = d.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = d.apply(lambda r: obi(r, 3), axis=1)

    # --- Funding Rates ---
    if 'funding_rate' in d.columns:
        fr = d['funding_rate']
        feat['funding_val'] = fr
        feat['funding_diff'] = fr.diff()
        feat['funding_8h_z'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-12)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    # Filter 0 variance or missing features
    feat = feat.dropna(axis=0)
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

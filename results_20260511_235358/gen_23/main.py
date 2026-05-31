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
    "max_depth": 3,              # Shallow trees generalize better across regimes
    "min_child_weight": 80,      # High weight prevents learning specific regime leaf nodes
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,           # L2 helps prevent extreme weights in the trees
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.03,       # Slow learning for better generalization
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # target metric requirement
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-invariant features using local z-score normalization
    and market efficiency metrics.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Fundamental price change
    log_ret = np.log(close / close.shift(1))

    def rolling_z(s, w=200):
        # Local normalization makes features regime-invariant
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)

    # --- Multi-Horizon Returns (Z-scored) ---
    for w in [4, 12, 48, 144]:
        ret = close.pct_change(w)
        feat[f'ret_z_{w}'] = rolling_z(ret, 400)

    # --- Market Efficiency & Trend Dynamics ---
    # Efficiency Ratio (Kaufman): displacement / total path
    # 1.0 = perfect trend, 0.0 = pure noise
    for w in [20, 60]:
        diff = (close - close.shift(w)).abs()
        path = close.diff().abs().rolling(w).sum()
        feat[f'efficiency_{w}'] = diff / (path + 1e-9)
    
    feat['efficiency_trend'] = feat['efficiency_20'].diff(10)

    # --- Volatility Dynamics ---
    vol_fast = log_ret.rolling(10).std()
    vol_slow = log_ret.rolling(60).std()
    feat['vol_ratio'] = vol_fast / (vol_slow + 1e-9)
    feat['vol_z_100'] = rolling_z(vol_fast, 200)
    feat['vol_accel'] = vol_fast.diff(5) / (vol_slow + 1e-9)

    # --- Relative Strength & Momentum ---
    for w in [14, 50]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        # Center RSI around 0 and normalize
        feat[f'rsi_norm_{w}'] = (rsi - 50) / 20.0

    # --- Mean Reversion / Positioning ---
    # Bollinger distance z-scored
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pct = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['bb_pct_z'] = rolling_z(bb_pct, 100)
    
    # Close relative to high-low range
    cl_pos = (close - low.rolling(30).min()) / (high.rolling(30).max() - low.rolling(30).min() + 1e-9)
    feat['range_pos_z'] = rolling_z(cl_pos, 100)

    # --- Volume Regime ---
    # Log volume usually works better with z-score
    log_vol = np.log(volume + 1e-9)
    feat['volume_z'] = rolling_z(log_vol, 200)
    feat['volume_spike'] = volume / (volume.rolling(50).mean() + 1e-9)
    
    # Volume-Price Correlation (identify distribution vs accumulation)
    feat['vol_price_corr'] = log_ret.rolling(20).corr(log_vol)

    # --- Interactions (The "Edge" features) ---
    # Momentum scaled by efficiency (only trust momentum in efficient trends)
    feat['efficient_mom'] = feat['ret_z_12'] * feat['efficiency_20']
    
    # Volatility-adjusted RSI
    feat['rsi_vol_interaction'] = feat['rsi_norm_14'] * feat['vol_ratio']

    # --- High-Low / Candle Character ---
    hl_range = (high - low) / (close + 1e-9)
    feat['hl_z'] = rolling_z(hl_range, 200)
    
    wick_ratio = (high - df[['open', 'close']].max(axis=1)) / (hl_range + 1e-9)
    feat['wick_ratio_z'] = rolling_z(wick_ratio, 100)

    # --- Time (Cyclical) ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # --- Funding Rate Regime ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        # Normalize funding by local vol to see if premium is "expensive"
        feat['funding_z'] = rolling_z(fr, 400)
        feat['funding_momentum'] = fr.diff(4)
        feat['funding_vol_adj'] = fr / (vol_slow + 1e-9)

    # Clean up
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.ffill().dropna()
    
    # Limit to robust features only
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
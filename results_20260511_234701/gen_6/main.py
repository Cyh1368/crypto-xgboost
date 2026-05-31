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
    "n_estimators": 220,
    "max_depth": 3,              # Shallower trees generalize better across regimes
    "min_child_weight": 15,      # High weight prevents overfitting to regime-specific noise
    "reg_alpha": 1.5,            # L1 regularization for feature selection
    "reg_lambda": 5.0,           # L2 regularization to prevent weight explosion
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 95          # Aggressive clipping to focus on robust mid-tails
MIN_PRED_STD_RATIO = 0.20     # Target higher prediction variance for better slope
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a regime-invariant feature set using rolling Z-scores 
    and volatility-normalized indicators.
    """
    feat = pd.DataFrame(index=df.index)
    
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # 1. Volatility - The fundamental denominator for regime invariance
    log_ret = np.log(close / close.shift(1))
    vol_fast = log_ret.rolling(20).std()
    vol_slow = log_ret.rolling(100).std()
    feat['vol_ratio'] = vol_fast / (vol_slow + 1e-9)
    
    # 2. Z-Score Normalized Returns (Regime Invariant Momentum)
    for lag in [1, 3, 8, 20]:
        ret = close.pct_change(lag)
        # Normalize return by its own rolling volatility
        feat[f'z_ret_{lag}'] = ret / (vol_fast * np.sqrt(lag) + 1e-9)
        # Rolling Z-score of returns
        role_mu = ret.rolling(60).mean()
        role_std = ret.rolling(60).std()
        feat[f'ret_zscore_{lag}'] = (ret - role_mu) / (role_std + 1e-9)

    # 3. Relative Range and Intraday Character
    hl_range = (high - low) / close
    feat['z_range'] = (hl_range - hl_range.rolling(40).mean()) / (hl_range.rolling(40).std() + 1e-9)
    
    # Candle wicks normalized by the total range
    candle_range = (high - low) + 1e-9
    feat['upper_wick_rel'] = (high - df[['open', 'close']].max(axis=1)) / candle_range
    feat['lower_wick_rel'] = (df[['open', 'close']].min(axis=1) - low) / candle_range
    feat['body_rel'] = (close - df['open']).abs() / candle_range

    # 4. Normalized Volume Dynamics
    # Volume relative to its recent average (Z-score)
    log_vol = np.log(volume + 1e-9)
    feat['z_volume'] = (log_vol - log_vol.rolling(40).mean()) / (log_vol.rolling(40).std() + 1e-9)
    feat['volume_trend'] = volume.rolling(5).mean() / (volume.rolling(40).mean() + 1e-9)

    # 5. Oscillator Scaling (RSI & Distance from Mean)
    for p in [14, 30]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        # Center RSI (0 to 1 -> -0.5 to 0.5)
        feat[f'rsi_norm_{p}'] = (rsi - 50) / 50.0

    # 6. Mean Reversion & Efficiency
    # Price distance from SMA normalized by volatility
    for p in [20, 50]:
        sma = close.rolling(p).mean()
        feat[f'dist_sma_{p}_z'] = (close - sma) / (close.rolling(p).std() + 1e-9)

    # Efficiency Ratio (Fractalness)
    net_change = (close - close.shift(20)).abs()
    sum_abs_change = close.diff().abs().rolling(20).sum()
    feat['efficiency_ratio'] = net_change / (sum_abs_change + 1e-9)

    # 7. Time Features (Cyclical)
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # 8. Acceleration (Rate of change of momentum)
    feat['mom_accel'] = feat['z_ret_1'].diff(3)
    
    # 9. Liquidity/Orderbook indicators (if present)
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['fr_zscore'] = (fr - fr.rolling(100).mean()) / (fr.rolling(100).std() + 1e-9)
        feat['fr_ret_interaction'] = feat['fr_zscore'] * feat['z_ret_3']

    # 10. Trend Consistency
    # Do signs of returns match over windows?
    signs = np.sign(log_ret)
    feat['trend_consistency'] = signs.rolling(10).mean()

    # Final cleanup
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    # Ensure feature count is within limits (< 80)
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
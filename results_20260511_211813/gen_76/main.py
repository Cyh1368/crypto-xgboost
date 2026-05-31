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
    "max_depth": 8,               # Deeper trees for complex regime interactions
    "min_child_weight": 1,        # Allow model to pick up small-sample signals
    "reg_alpha": 0.0,             # Minimize L1 to prevent signal suppression
    "reg_lambda": 0.001,          # Minimal L2 to allow variance in predictions
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.08,        # Higher learning rate helps avoid mean collapse
    "gamma": 0.05,                # Light pruning for stability
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 60,
}

CLIP_PERCENTILE = 95          # Focus on the 'meat' of the distribution
MIN_PRED_STD_RATIO = 0.15     # Aggressive target for prediction dispersion
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix focusing on microstructure and regime-conditional exhaustion.
    Max 80 features. Strictly no lookahead (shift(1)).
    """
    feat = pd.DataFrame(index=df.index)

    # Shift base series to ensure no lookahead
    # All features are computed based on 'df_past'
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)
    o = df['open'].shift(1)

    # 1. Momentum & Scale-Invariant Returns
    log_ret = np.log(c / c.shift(1))
    for lag in [1, 3, 6, 12, 24, 48]:
        feat[f'ret_{lag}'] = c.pct_change(lag)
    
    # Return Acceleration
    feat['accel_3_6'] = feat['ret_3'] - feat['ret_6']
    
    # 2. Volatility Microstructure
    vol5 = log_ret.rolling(5).std()
    vol20 = log_ret.rolling(20).std()
    feat['vol_ratio'] = vol5 / (vol20 + 1e-9)
    feat['vol_z'] = (vol20 - vol20.rolling(100).mean()) / (vol20.rolling(100).std() + 1e-9)
    
    # 3. Price Velocity (Returns normalized by Vol)
    # This helps find 'sharp' moves vs 'noisy' moves
    feat['speed_1'] = log_ret / (vol5 + 1e-9)
    feat['speed_6'] = c.pct_change(6) / (vol20 + 1e-9)

    # 4. Exhaustion & Range Features
    # Bars since N-period High/Low
    feat['bars_since_max_20'] = c.rolling(20).apply(lambda x: 20 - np.argmax(x) if len(x) > 0 else 0)
    feat['bars_since_min_20'] = c.rolling(20).apply(lambda x: 20 - np.argmin(x) if len(x) > 0 else 0)
    
    # Efficiency: net move vs absolute move (Efficiency Ratio)
    abs_move_sum = log_ret.abs().rolling(10).sum()
    net_move = (c - c.shift(10)).abs()
    feat['efficiency_10'] = net_move / (abs_move_sum * c + 1e-9)

    # 5. Candlestick Patterns (Normalized)
    feat['body_pct'] = (c - o) / (h - l + 1e-9)
    feat['upper_wick_pct'] = (h - np.maximum(c, o)) / (h - l + 1e-9)
    feat['lower_wick_pct'] = (np.minimum(c, o) - l) / (h - l + 1e-9)
    feat['range_to_vol'] = (h - l) / (c * vol20 + 1e-9)

    # 6. Trend & Deviation
    sma20 = c.rolling(20).mean()
    sma100 = c.rolling(100).mean()
    feat['z_score_20'] = (c - sma20) / (c.rolling(20).std() + 1e-9)
    feat['dist_sma100'] = (c - sma100) / (sma100 + 1e-9)
    feat['sma_trend'] = (sma20 > sma100).astype(float)
    
    # 7. Volume Intensity
    v_ma20 = v.rolling(20).mean()
    feat['volu_z'] = (v - v_ma20) / (v.rolling(20).std() + 1e-9)
    feat['volu_shock'] = v / (v.rolling(5).mean() + 1e-9)
    feat['pvt'] = (log_ret * v).rolling(10).sum() / v_ma20 # simplified price-volume trend

    # 8. Oscillators
    def get_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    feat['rsi_7'] = get_rsi(c, 7)
    feat['rsi_14'] = get_rsi(c, 14)
    feat['rsi_div'] = feat['rsi_7'] - feat['rsi_14']

    # 9. Regime Indicators (Binary for tree splits)
    feat['high_vol_regime'] = (vol20 > vol20.rolling(200).mean()).astype(float)
    feat['uptrend_regime'] = (c > sma20).astype(float)
    
    # Interactions
    feat['rsi_high_vol'] = feat['rsi_14'] * feat['high_vol_regime']
    feat['mom_trend_align'] = np.sign(feat['ret_6']) * feat['sma_trend']

    # 10. Funding & Orderbook (if available)
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].shift(1)
        feat['funding_val'] = fr
        feat['funding_z'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)
    
    if 'bids' in df.columns and 'asks' in df.columns:
        # Simple mid-price spread / imbalance from past column
        def fast_obi(row):
            try:
                b = row['bids'][0][1]
                a = row['asks'][0][1]
                return (b - a) / (b + a + 1e-9)
            except: return 0.0
        feat['obi_raw'] = df.shift(1).apply(fast_obi, axis=1)

    # 11. Time Features
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    # Clean and fill
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    
    # Final check on feature count
    # print(f"Feature count: {len(feat.columns)}")
    
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
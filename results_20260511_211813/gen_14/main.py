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
    "max_depth": 5,
    "min_child_weight": 2,      # Reduced to increase expressivity
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "learning_rate": 0.04,      # Faster learning to capture signal
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 99          # clip targets at 99th percentile
MIN_PRED_STD_RATIO = 0.15     # target for prediction variance
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with modular components: 
    Returns, Volatility, Momentum, Volume, and Micro-Structure.
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    open_p = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. Multi-Scale Returns & Momentum
    for lag in [1, 3, 6, 12, 48]:
        feat[f'ret_{lag}'] = close.pct_change(lag)
    
    feat['mom_accel'] = feat['ret_1'] - feat['ret_3'] / 3.0
    
    # 2. Volatility & Risk
    log_ret = np.log(close / close.shift(1))
    feat['vol_short'] = log_ret.rolling(10).std()
    feat['vol_long'] = log_ret.rolling(50).std()
    feat['vol_regime'] = feat['vol_short'] / (feat['vol_long'] + 1e-9)
    
    # Parkinson Volatility (High-Low range based)
    feat['vol_parkinson'] = np.sqrt(1 / (4 * np.log(2)) * (np.log(high/low)**2).rolling(20).mean())

    # 3. Micro-Price Structure
    candle_range = (high - low).replace(0, 1e-9)
    feat['body_pct'] = (close - open_p) / candle_range
    feat['upper_wick'] = (high - np.maximum(close, open_p)) / candle_range
    feat['lower_wick'] = (np.minimum(close, open_p) - low) / candle_range
    
    # Price location in 20-period range
    roll_low = low.rolling(20).min()
    roll_high = high.rolling(20).max()
    feat['range_pos'] = (close - roll_low) / (roll_high - roll_low + 1e-9)

    # 4. Momentum Indicators
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat['rsi_14'] = 100 - 100 / (1 + rs)
    
    # Bollinger Band Width
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_width'] = (bb_std * 4) / (bb_mid + 1e-9)
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # 5. Volume Dynamics
    feat['vol_zscore_20'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    feat['vol_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    
    # Price-Volume Trend (Simplified)
    feat['pv_trend'] = feat['ret_1'] * feat['vol_ratio_5']

    # 6. Time Features
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # 7. Orderbook & Funding (Conditional)
    if 'bids' in df.columns and 'asks' in df.columns:
        def get_obi(row):
            try:
                b_vol = sum(x[1] for x in row['bids'][:5])
                a_vol = sum(x[1] for x in row['asks'][:5])
                return (b_vol - a_vol) / (b_vol + a_vol + 1e-9)
            except: return 0.0
        feat['obi_simple'] = df.apply(get_obi, axis=1)

    if 'funding_rate' in df.columns:
        feat['funding_rate'] = df['funding_rate']
        feat['funding_ma'] = df['funding_rate'].rolling(32).mean()

    # Interaction: Volatility-adjusted momentum
    feat['sharpe_12'] = close.pct_change(12) / (log_ret.rolling(12).std() + 1e-9)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()
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

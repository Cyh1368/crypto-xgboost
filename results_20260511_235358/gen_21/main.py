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
    "min_child_weight": 4,
    "reg_alpha": 0.05,
    "reg_lambda": 0.8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # Clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # Minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a regime-invariant feature set by applying rolling Z-score 
    standardization to all major market indicators. 
    """
    feat_blocks = []
    
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    open_ = df['open']
    
    log_ret = np.log(close / close.shift(1))
    
    # 1. Block: Momentum (Rolling Z-Scores of Returns)
    # This captures how 'extreme' the current return is compared to the recent past.
    mom_feat = pd.DataFrame(index=df.index)
    for lag in [1, 3, 5, 10]:
        ret = close.pct_change(lag)
        mom_feat[f'ret_z_{lag}'] = (ret - ret.rolling(20).mean()) / (ret.rolling(20).std() + 1e-9)
    feat_blocks.append(mom_feat)
    
    # 2. Block: Volatility & Vol-Regime
    # Volatility is normalized to make the model regime-blind to absolute vol levels.
    vol_feat = pd.DataFrame(index=df.index)
    vols = {}
    for w in [5, 15, 30]:
        vols[w] = log_ret.rolling(w).std()
        vol_feat[f'vol_{w}_z'] = (vols[w] - vols[w].rolling(60).mean()) / (vols[w].rolling(60).std() + 1e-9)
    
    vol_feat['vol_ratio_fast'] = vols[5] / (vols[15] + 1e-9)
    vol_feat['vol_ratio_slow'] = vols[15] / (vols[30] + 1e-9)
    feat_blocks.append(vol_feat)
    
    # 3. Block: Price Position (Normalized Mean Reversion)
    # Using Z-scores of price deviations from its mean.
    pos_feat = pd.DataFrame(index=df.index)
    for w in [10, 20]:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        pos_feat[f'price_z_{w}'] = (close - ma) / (std + 1e-9)
        
    low_w = low.rolling(20).min()
    high_w = high.rolling(20).max()
    pos_feat['range_pos'] = (close - low_w) / (high_w - low_w + 1e-9)
    feat_blocks.append(pos_feat)
    
    # 4. Block: Oscillators (Short Lookbacks)
    osc_feat = pd.DataFrame(index=df.index)
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        osc_feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)
    feat_blocks.append(osc_feat)
    
    # 5. Block: Market Character (Efficiency & Autocorr)
    char_feat = pd.DataFrame(index=df.index)
    # Efficiency ratio measures "trendiness" (regime invariant)
    char_feat['eff_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    char_feat['autocorr_10'] = log_ret.rolling(10).corr(log_ret.shift(1))
    feat_blocks.append(char_feat)
    
    # 6. Block: Volume Dynamics
    volu_feat = pd.DataFrame(index=df.index)
    volu_feat['vol_v_ma5'] = volume / (volume.rolling(5).mean() + 1e-9)
    vol_ma20 = volume.rolling(20).mean()
    volu_feat['vol_z_20'] = (volume - vol_ma20) / (volume.rolling(20).std() + 1e-9)
    feat_blocks.append(volu_feat)
    
    # 7. Block: Time of Day
    time_feat = pd.DataFrame(index=df.index)
    time_feat['hr_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    time_feat['hr_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    time_feat['is_we'] = (df.index.dayofweek >= 5).astype(float)
    feat_blocks.append(time_feat)
    
    # 8. Block: Funding (Normalized)
    if 'funding_rate' in df.columns:
        fund_feat = pd.DataFrame(index=df.index)
        fr = df['funding_rate']
        fund_feat['fr_z'] = (fr - fr.rolling(32).mean()) / (fr.rolling(32).std() + 1e-9)
        fund_feat['fr_diff'] = fr.diff(4) # Momentum in funding
        feat_blocks.append(fund_feat)
        
    # 9. Block: Interactions (Momentum X Character)
    # These often help identify when a regime shift is occurring
    inter_feat = pd.DataFrame(index=df.index)
    ret_z = (close.pct_change(3) - close.pct_change(3).rolling(20).mean()) / (close.pct_change(3).rolling(20).std() + 1e-9)
    inter_feat['mom_vol_inter'] = ret_z * vol_feat['vol_5_z']
    inter_feat['mom_eff_inter'] = ret_z * char_feat['eff_10']
    feat_blocks.append(inter_feat)

    # Combine all blocks
    feat = pd.concat(feat_blocks, axis=1)
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
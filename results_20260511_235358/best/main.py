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
    "n_estimators": 300,
    "max_depth": 4,
    "min_child_weight": 15,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.25     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with regime-invariant rolling Z-scores.
    Focuses on shorter lookbacks to minimize decay between regimes.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df['volume']
    high, low = df['high'], df['low']

    # --- Returns normalized by rolling volatility ---
    log_ret = np.log(close / close.shift(1))
    vol_fast = log_ret.rolling(15).std()
    vol_slow = log_ret.rolling(50).std()

    for lag in [1, 2, 3, 5, 10, 20]:
        # Divide returns by annualized-style rolling volatility to make features regime-invariant
        feat[f'ret_z_{lag}'] = (close.pct_change(lag)) / (vol_fast * np.sqrt(lag) + 1e-9)

    # --- Volatility Dynamics ---
    feat['vol_relative_ratio'] = vol_fast / (vol_slow + 1e-9)
    feat['vol_change'] = vol_fast.pct_change(5)
    feat['vol_z'] = (vol_fast - vol_fast.rolling(50).mean()) / (vol_fast.rolling(50).std() + 1e-9)

    # --- RSI (Z-scored) ---
    for period in [7, 14]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        # Normalize RSI to locally invariant Z-score
        feat[f'rsi_z_{period}'] = (rsi - rsi.rolling(40).mean()) / (rsi.rolling(40).std() + 1e-9)

    # --- Bollinger / VWAP Deviations (Z-scored) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pct = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['bb_pct_z'] = (bb_pct - bb_pct.rolling(20).mean()) / (bb_pct.rolling(20).std() + 1e-9)

    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    vwap_dev = (close - vwap) / (close + 1e-9)
    feat['vwap_dev_z'] = (vwap_dev - vwap_dev.rolling(20).mean()) / (vwap_dev.rolling(20).std() + 1e-9)

    # --- Efficiency and Trend ---
    # Efficiency ratio: absolute move / sum of moves.
    er_15 = (close - close.shift(15)).abs() / (close.diff().abs().rolling(15).sum() + 1e-9)
    feat['efficiency_15'] = er_15
    feat['er_trend'] = er_15 - er_15.rolling(10).mean()

    # --- Volume Dynamics (Log-normalized) ---
    feat['volume_z'] = np.log(volume / (volume.rolling(40).mean() + 1e-9) + 1e-9)
    feat['volume_adv_ratio'] = volume / (volume.rolling(100).mean() + 1e-9)

    # --- Time Features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Mean Reversion ---
    sma_10 = close.rolling(10).mean()
    price_dev = (close - sma_10) / (sma_10 + 1e-9)
    feat['price_dev_z'] = (price_dev - price_dev.rolling(20).mean()) / (price_dev.rolling(20).std() + 1e-9)

    # --- Momentum x Volatility Interaction ---
    # Captures "high vol breakouts" vs "low vol grind"
    feat['mom_vol_interaction'] = feat['ret_z_3'] * feat['vol_relative_ratio']

    # --- Candlestick Shape ---
    feat['hl_range_z'] = (high - low) / (close * vol_fast + 1e-9)
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)

    # --- Momentum Consistency ---
    feat['autocorr_5'] = log_ret.rolling(10).corr(log_ret.shift(1))

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_z'] = (fr - fr.rolling(40).mean()) / (fr.rolling(40).std() + 1e-9)
        feat['funding_mom'] = fr.diff(4) / (vol_fast + 1e-9)

    # --- Orderbook ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                b_v = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                a_v = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (b_v - a_v) / (b_v + a_v + 1e-9)
            except Exception: return 0.0
        feat['obi_z'] = df.apply(lambda r: obi(r, 1), axis=1)

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
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
    "min_child_weight": 12,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.15     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    Uses rolling Z-scores and regime-invariant indicators to improve generalization.
    """
    feat = pd.DataFrame(index=df.index)
    close, volume, high, low = df['close'], df['volume'], df['high'], df['low']
    log_ret = np.log(close / close.shift(1))

    # --- Returns (Fibonacci lags) ---
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    # --- Volatility (Z-scored) ---
    vol_10 = log_ret.rolling(10).std()
    feat['vol_10'] = vol_10
    feat['vol_z_60'] = (vol_10 - vol_10.rolling(60).mean()) / (vol_10.rolling(60).std() + 1e-9)
    feat['vol_trend'] = vol_10.diff(5)

    # --- RSI (Z-scored) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_14 = 100 - 100 / (1 + gain / (loss + 1e-9))
    feat['rsi_14'] = rsi_14
    feat['rsi_z_60'] = (rsi_14 - rsi_14.rolling(60).mean()) / (rsi_14.rolling(60).std() + 1e-9)

    # --- Price Deviation (Z-scored) ---
    sma_20 = close.rolling(20).mean()
    price_dev_20 = (close - sma_20) / (sma_20 + 1e-9)
    feat['price_dev_z_60'] = (price_dev_20 - price_dev_20.rolling(60).mean()) / (price_dev_20.rolling(60).std() + 1e-9)
    feat['sma_cross'] = (close.rolling(10).mean() - sma_20) / (sma_20 + 1e-9)

    # --- Regime Indicators ---
    feat['vol_regime'] = vol_10 / (vol_10.rolling(80).mean() + 1e-9)
    feat['vol_rank'] = vol_10.rolling(80).rank(pct=True)
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)

    # --- Momentum & Interactions ---
    ret_3 = close.pct_change(3)
    feat['momentum_regime'] = ret_3 / (vol_10 + 1e-9)
    feat['mom_vol_interaction'] = ret_3 * feat['vol_regime']
    feat['volume_price_pressure'] = (volume / (volume.rolling(20).mean() + 1e-9)) * price_dev_20

    # --- Technical Indicators ---
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - sma_20) / (2 * bb_std + 1e-9)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat['atr_norm'] = tr.rolling(14).mean() / close
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    feat['macd_signal'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

    # --- Volume & Correlation ---
    feat['volume_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    feat['vol_ret_corr_20'] = log_ret.rolling(20).corr(volume.pct_change(1))

    # --- Range & Candle Structure ---
    feat['hl_range'] = (high - low) / close
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['close_position_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-9)

    # --- Time Features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Orderbook & Funding ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids, asks = np.array(row['bids']), np.array(row['asks'])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except: return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)

    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_z'] = (fr - fr.rolling(60).mean()) / (fr.rolling(60).std() + 1e-9)
        feat['funding_momentum'] = fr.diff(4)

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
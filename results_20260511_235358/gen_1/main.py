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
    "n_estimators": 250,
    "max_depth": 4,
    "min_child_weight": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

CLIP_PERCENTILE = 98          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.15     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features must use only past data (no lookahead).
    Returns a DataFrame aligned to df.index.

    Key strategy: Use shorter lookback windows (5-20 bars) and z-score
    normalization within rolling windows to make features regime-invariant.
    Add rate-of-change features to capture evolving market character.
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    # --- Returns (shorter lags for faster adaptation) ---
    for lag in [1, 2, 3, 6, 12]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    # --- Volatility (shorter windows + z-score normalization) ---
    log_ret = np.log(close / close.shift(1))
    for w in [5, 10, 15]:
        vol_w = log_ret.rolling(w).std()
        feat[f'vol_{w}'] = vol_w
        # Z-score normalize volatility within a rolling window (regime-invariant)
        vol_mean = vol_w.rolling(30).mean()
        vol_std = vol_w.rolling(30).std()
        feat[f'vol_{w}_zscore'] = (vol_w - vol_mean) / (vol_std + 1e-9)

    # --- Volatility trend (rate of change of volatility) ---
    vol_15 = log_ret.rolling(15).std()
    feat['vol_trend_5'] = vol_15.diff(5) / (vol_15.shift(5) + 1e-9)
    feat['vol_trend_10'] = vol_15.diff(10) / (vol_15.shift(10) + 1e-9)

    # --- RSI (shorter periods) ---
    for period in [6, 12]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands (shorter window) ---
    bb_mid = close.rolling(15).mean()
    bb_std = close.rolling(15).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    # Z-score the BB position within rolling window
    bb_pct_mean = feat['bb_pct'].rolling(30).mean()
    bb_pct_std = feat['bb_pct'].rolling(30).std()
    feat['bb_pct_zscore'] = (feat['bb_pct'] - bb_pct_mean) / (bb_pct_std + 1e-9)

    # --- ATR (shorter window) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_10'] = tr.rolling(10).mean()
    feat['atr_norm'] = feat['atr_10'] / close

    # --- MACD (shorter spans for faster response) ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    macd = ema8 - ema17
    feat['macd_signal'] = macd - macd.ewm(span=6, adjust=False).mean()

    # --- Volume (shorter windows) ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_10'] = volume / (volume.rolling(10).mean() + 1e-9)
    vwap = (close * volume).rolling(15).sum() / (volume.rolling(15).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)
    # Volume trend
    vol_ma_5 = volume.rolling(5).mean()
    feat['volume_trend'] = vol_ma_5.diff(5) / (vol_ma_5.shift(5) + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features (shorter lookbacks) ---
    feat['trend_strength_10'] = (close / close.shift(10) - 1).abs() * 10000
    feat['trend_strength_15'] = (close / close.shift(15) - 1).abs() * 10000
    trend_10 = (close / close.shift(10) - 1) * 10000
    trend_15 = (close / close.shift(15) - 1) * 10000
    feat['trend_dir_10'] = np.sign(trend_10)
    feat['trend_dir_15'] = np.sign(trend_15)

    # Volatility regime (shorter base window, faster adaptation)
    realized_vol_10 = log_ret.rolling(10).std()
    realized_vol_15 = log_ret.rolling(15).std()
    feat['vol_regime_10'] = realized_vol_10 / (realized_vol_10.rolling(60).mean() + 1e-9)
    feat['vol_regime_fast'] = realized_vol_10 / (realized_vol_15 + 1e-9)
    feat['vol_rank_60'] = realized_vol_10.rolling(60).rank(pct=True)

    # --- Mean reversion features (shorter windows) ---
    sma_5 = close.rolling(5).mean()
    sma_15 = close.rolling(15).mean()
    feat['price_sma5_dev'] = (close - sma_5) / (sma_5 + 1e-9)
    feat['price_sma15_dev'] = (close - sma_15) / (sma_15 + 1e-9)
    feat['sma_cross'] = (sma_5 - sma_15) / (sma_15 + 1e-9)
    # Mean reversion strength (z-scored)
    mr_mean = feat['price_sma5_dev'].rolling(30).mean()
    mr_std = feat['price_sma5_dev'].rolling(30).std()
    feat['price_sma5_dev_zscore'] = (feat['price_sma5_dev'] - mr_mean) / (mr_std + 1e-9)

    # --- Momentum × Volatility interactions (regime-aware) ---
    ret_2 = close.pct_change(2)
    vol_10 = log_ret.rolling(10).std()
    feat['momentum_vol_interaction'] = ret_2 * vol_10
    feat['momentum_regime'] = ret_2 / (vol_10 + 1e-9)
    feat['momentum_regime_10'] = ret_2 * feat['vol_regime_10']
    feat['momentum_regime_fast'] = ret_2 * feat['vol_regime_fast']

    # --- Volume × Price position interactions ---
    vol_ratio = volume / (volume.rolling(10).mean() + 1e-9)
    bb_pct = (close - close.rolling(15).mean()) / (2 * close.rolling(15).std() + 1e-9)
    feat['volume_bb_interaction'] = vol_ratio * bb_pct
    feat['volume_price_pressure'] = vol_ratio * feat['price_sma5_dev']

    # --- High-low range features ---
    feat['hl_range'] = (high - low) / close
    feat['hl_range_ma'] = feat['hl_range'].rolling(8).mean()
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Close position in range (shorter window) ---
    low_15 = low.rolling(15).min()
    high_15 = high.rolling(15).max()
    feat['close_position_15'] = (close - low_15) / (high_15 - low_15 + 1e-9)

    # --- Return z-scores (regime-invariant momentum) ---
    ret_1 = close.pct_change(1)
    ret_1_mean = ret_1.rolling(15).mean()
    ret_1_std = ret_1.rolling(15).std()
    feat['ret_z_15'] = (ret_1 - ret_1_mean) / (ret_1_std + 1e-9)

    feat['ret_accel_4'] = close.pct_change(1).diff(4)
    feat['vol_ret_corr_15'] = log_ret.rolling(15).corr(volume.pct_change(1))

    # --- Autocorrelation trend (how persistent is momentum) ---
    feat['ret_autocorr_5'] = ret_1.rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
    feat['ret_autocorr_trend'] = feat['ret_autocorr_5'].diff(5)

    # --- Orderbook (if real data available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_4h_ma'] = fr.rolling(16).mean()   # 16 × 15min = 4h
        feat['funding_momentum'] = fr.diff(2)
        feat['funding_x_volregime'] = fr * feat['vol_regime_10']

    # --- Efficiency ratio (shorter window) ---
    feat['efficiency_15'] = (close - close.shift(15)).abs() / (close.diff().abs().rolling(15).sum() + 1e-9)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
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
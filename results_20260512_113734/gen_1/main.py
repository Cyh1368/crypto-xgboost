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
    "max_depth": 5,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.15,
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
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    # --- Returns with z-score normalization ---
    log_ret = np.log(close / close.shift(1))
    for lag in [1, 2, 3, 6, 12]:
        ret_lag = close.pct_change(lag)
        # Z-score normalize within rolling 20-bar window
        ret_mean = ret_lag.rolling(20).mean()
        ret_std = ret_lag.rolling(20).std()
        feat[f'ret_{lag}_z'] = (ret_lag - ret_mean) / (ret_std + 1e-9)

    # --- Volatility with shorter lookbacks and z-score ---
    for w in [5, 10, 20]:
        vol = log_ret.rolling(w).std()
        feat[f'vol_{w}'] = vol
        # Volatility z-score relative to recent history
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        feat[f'vol_{w}_z'] = (vol - vol_mean) / (vol_std + 1e-9)

    # Volatility trend (rate of change)
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_trend'] = (vol_5 - vol_20) / (vol_20 + 1e-9)

    # --- RSI (shorter periods, normalized) ---
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        # Normalize RSI to [-1, 1] range for regime invariance
        feat[f'rsi_{period}'] = (rsi - 50) / 50

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm'] = feat['atr_14'] / close

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feat['macd_signal'] = macd - macd.ewm(span=9, adjust=False).mean()

    # --- Volume ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features (shorter lookbacks, z-scored) ---
    # Use 10 and 20 bar trends instead of 20 and 60
    ret_10 = close.pct_change(10)
    ret_20 = close.pct_change(20)

    # Z-score normalize trends
    ret_10_mean = ret_10.rolling(20).mean()
    ret_10_std = ret_10.rolling(20).std()
    feat['trend_10_z'] = (ret_10 - ret_10_mean) / (ret_10_std + 1e-9)

    ret_20_mean = ret_20.rolling(20).mean()
    ret_20_std = ret_20.rolling(20).std()
    feat['trend_20_z'] = (ret_20 - ret_20_mean) / (ret_20_std + 1e-9)

    # Volatility regime using shorter windows
    realized_vol_10 = log_ret.rolling(10).std()
    realized_vol_20 = log_ret.rolling(20).std()
    feat['vol_regime_fast'] = realized_vol_10 / (realized_vol_20 + 1e-9)

    # Volatility rank over shorter window (60 instead of 200)
    feat['vol_rank_60'] = realized_vol_10.rolling(60).rank(pct=True)

    # --- Mean reversion features (shorter windows, z-scored) ---
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()

    # Price deviation from MA, z-scored
    dev_5 = (close - sma_5) / (sma_5 + 1e-9)
    dev_5_mean = dev_5.rolling(20).mean()
    dev_5_std = dev_5.rolling(20).std()
    feat['price_sma5_dev_z'] = (dev_5 - dev_5_mean) / (dev_5_std + 1e-9)

    dev_20 = (close - sma_20) / (sma_20 + 1e-9)
    dev_20_mean = dev_20.rolling(20).mean()
    dev_20_std = dev_20.rolling(20).std()
    feat['price_sma20_dev_z'] = (dev_20 - dev_20_mean) / (dev_20_std + 1e-9)

    # MA cross
    feat['sma_cross'] = (sma_5 - sma_20) / (sma_20 + 1e-9)

    # --- Momentum × Volatility interactions (z-scored) ---
    ret_3 = close.pct_change(3)
    vol_10 = log_ret.rolling(10).std()

    # Z-score the momentum first
    ret_3_mean = ret_3.rolling(20).mean()
    ret_3_std = ret_3.rolling(20).std()
    ret_3_z = (ret_3 - ret_3_mean) / (ret_3_std + 1e-9)

    # Interaction with volatility regime
    feat['momentum_vol_interaction'] = ret_3_z * feat['vol_regime_fast']

    # Risk-adjusted momentum
    feat['momentum_regime'] = ret_3_z / (vol_10 + 1e-9)

    # --- Volume features (z-scored) ---
    vol_ratio_10 = volume / (volume.rolling(10).mean() + 1e-9)
    vol_ratio_mean = vol_ratio_10.rolling(20).mean()
    vol_ratio_std = vol_ratio_10.rolling(20).std()
    feat['volume_ratio_z'] = (vol_ratio_10 - vol_ratio_mean) / (vol_ratio_std + 1e-9)

    # Volume trend (rate of change)
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    feat['volume_trend'] = (vol_ma_5 - vol_ma_20) / (vol_ma_20 + 1e-9)

    # --- High-low range features ---
    feat['hl_range'] = (high - low) / close
    feat['hl_range_ma'] = feat['hl_range'].rolling(10).mean()
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Close position in range (shorter window) ---
    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    feat['close_position_10'] = (close - low_10) / (high_10 - low_10 + 1e-9)

    # Return z-score (already good, keep one)
    feat['ret_z_20'] = (close.pct_change(1) - close.pct_change(1).rolling(20).mean()) / (close.pct_change(1).rolling(20).std() + 1e-9)

    # Return acceleration (shorter lag)
    ret_1 = close.pct_change(1)
    feat['ret_accel_3'] = ret_1.diff(3)

    # Autocorrelation as regime indicator
    feat['ret_autocorr_10'] = log_ret.rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)

    # Correlation between returns and volume
    feat['vol_ret_corr_10'] = log_ret.rolling(10).corr(volume.pct_change(1))

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
        feat['funding_8h_ma'] = fr.rolling(32).mean()   # 32 × 15min = 8h
        feat['funding_momentum'] = fr.diff(4)
        feat['funding_x_volregime'] = fr * feat['vol_regime']

    # --- Trend Consistency and Efficiency ---
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)

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
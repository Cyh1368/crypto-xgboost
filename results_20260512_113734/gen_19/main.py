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
    "n_estimators": 240,
    "max_depth": 4,
    "min_child_weight": 5,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "subsample": 0.78,
    "colsample_bytree": 0.75,
    "learning_rate": 0.08,
    "gamma": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 97          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.18     # minimum acceptable pred_std / actual_std
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
    eps = 1e-9

    # --- Returns (shorter horizons only) ---
    ret_1 = close.pct_change(1)
    feat['ret_1'] = ret_1
    for lag in [2, 3, 5, 8, 12, 20]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    # --- Rolling z-scored returns (regime-normalized) ---
    for w in [5, 10, 20]:
        mu = ret_1.rolling(w).mean()
        sd = ret_1.rolling(w).std()
        feat[f'ret_z_{w}'] = (ret_1 - mu) / (sd + eps)

    # --- Volatility (shorter windows) ---
    log_ret = np.log(close / close.shift(1))
    rv_5 = log_ret.rolling(5).std()
    rv_10 = log_ret.rolling(10).std()
    rv_20 = log_ret.rolling(20).std()

    feat['vol_5'] = rv_5
    feat['vol_10'] = rv_10
    feat['vol_20'] = rv_20

    # Volatility trend (rate of change)
    feat['vol_trend_5_10'] = (rv_5 - rv_10) / (rv_10 + eps)
    feat['vol_trend_10_20'] = (rv_10 - rv_20) / (rv_20 + eps)

    # --- RSI (shorter periods) ---
    delta = close.diff()
    for period in [5, 10, 14]:
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands (z-score style) ---
    for w in [10, 20]:
        bb_mid = close.rolling(w).mean()
        bb_std = close.rolling(w).std()
        feat[f'bb_z_{w}'] = (close - bb_mid) / (bb_std + eps)

    # --- ATR (shorter window, normalized) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_10 = tr.rolling(10).mean()
    feat['atr_norm'] = atr_10 / (close + eps)
    feat['atr_change'] = atr_10.pct_change(5)

    # --- MACD (shorter spans) ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd = ema8 - ema21
    feat['macd'] = macd / (close + eps)
    feat['macd_signal'] = (macd - macd.ewm(span=5, adjust=False).mean()) / (close + eps)

    # --- Volume (z-scored) ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()

    feat['volume_ratio_5'] = volume / (vol_ma_5 + eps)
    feat['volume_z_20'] = (volume - vol_ma_20) / (vol_std_20 + eps)

    vwap_10 = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + eps)
    vwap_20 = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + eps)
    feat['vwap_dev_10'] = (close - vwap_10) / (close + eps)
    feat['vwap_dev_20'] = (close - vwap_20) / (close + eps)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features (shorter windows, rate-of-change focus) ---
    # Trend at multiple short horizons
    for w in [5, 10, 20]:
        feat[f'trend_{w}'] = close / close.shift(w) - 1

    # Efficiency ratio (trend strength relative to noise)
    for w in [10, 20]:
        net_move = (close - close.shift(w)).abs()
        total_move = close.diff().abs().rolling(w).sum()
        feat[f'efficiency_{w}'] = net_move / (total_move + eps)

    # Volatility regime (shorter baseline)
    vol_base_60 = rv_20.rolling(60).mean()
    feat['vol_regime'] = rv_20 / (vol_base_60 + eps)
    feat['vol_regime_change'] = feat['vol_regime'].diff(3)

    # Autocorrelation trend (persistence change)
    feat['ret_autocorr_5'] = ret_1.rolling(5).corr(ret_1.shift(1))
    feat['ret_autocorr_10'] = ret_1.rolling(10).corr(ret_1.shift(1))
    feat['autocorr_trend'] = feat['ret_autocorr_5'] - feat['ret_autocorr_10']

    # --- Mean reversion features (shorter windows) ---
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()

    feat['price_sma5_dev'] = (close - sma_5) / (sma_5 + eps)
    feat['price_sma10_dev'] = (close - sma_10) / (sma_10 + eps)
    feat['price_sma20_dev'] = (close - sma_20) / (sma_20 + eps)
    feat['sma5_sma20_cross'] = (sma_5 - sma_20) / (sma_20 + eps)

    # --- Momentum × Volatility interactions ---
    ret_3 = close.pct_change(3)
    vol_20 = log_ret.rolling(20).std()
    feat['momentum_vol_interaction'] = ret_3 * vol_20
    feat['momentum_regime'] = ret_3 / (vol_20 + 1e-9)
    feat['momentum_regime_20'] = ret_3 * feat['vol_regime']
    feat['momentum_regime_fast'] = ret_3 * feat['vol_regime_fast']

    # --- Volume × Price position interactions ---
    vol_ratio = volume / (volume.rolling(20).mean() + 1e-9)
    bb_pct = (close - close.rolling(20).mean()) / (2 * close.rolling(20).std() + 1e-9)
    feat['volume_bb_interaction'] = vol_ratio * bb_pct
    feat['volume_price_pressure'] = vol_ratio * feat['price_sma10_dev']

    # --- High-low range features ---
    feat['hl_range'] = (high - low) / close
    feat['hl_range_ma'] = feat['hl_range'].rolling(10).mean()
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Close position in range ---
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['close_position_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)
    feat['ret_z_20'] = (close.pct_change(1) - close.pct_change(1).rolling(20).mean()) / (close.pct_change(1).rolling(20).std() + 1e-9)
    feat['ret_z_60'] = (close.pct_change(1) - close.pct_change(1).rolling(60).mean()) / (close.pct_change(1).rolling(60).std() + 1e-9)
    feat['ret_accel_6'] = close.pct_change(1).diff(6)
    feat['vol_ret_corr_20'] = log_ret.rolling(20).corr(volume.pct_change(1))

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
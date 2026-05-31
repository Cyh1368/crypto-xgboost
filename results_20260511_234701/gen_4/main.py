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
    "n_estimators": 260,
    "max_depth": 3,
    "min_child_weight": 8,
    "reg_alpha": 0.4,
    "reg_lambda": 2.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

CLIP_PERCENTILE = 99          # clip targets at this percentile (both tails)
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

    # --- Returns ---
    log_ret = np.log(close / close.shift(1))
    short_lags = [1, 2, 3, 5, 8, 12, 20]
    for lag in short_lags:
        feat[f'ret_{lag}'] = close.pct_change(lag)
    feat['ret_1_z_20'] = (close.pct_change(1) - close.pct_change(1).rolling(20).mean()) / (close.pct_change(1).rolling(20).std() + 1e-9)
    feat['ret_3_z_20'] = (close.pct_change(3) - close.pct_change(3).rolling(20).mean()) / (close.pct_change(3).rolling(20).std() + 1e-9)

    # --- Volatility ---
    for w in [5, 10, 20, 40]:
        vol_w = log_ret.rolling(w).std()
        feat[f'vol_{w}'] = vol_w
        feat[f'vol_z_{w}'] = (vol_w - vol_w.rolling(60).mean()) / (vol_w.rolling(60).std() + 1e-9)

    # --- RSI ---
    for period in [5, 10]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}'] = rsi
        feat[f'rsi_z_{period}'] = (rsi - rsi.rolling(40).mean()) / (rsi.rolling(40).std() + 1e-9)

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)
    feat['bb_pct_z_40'] = (feat['bb_pct'] - feat['bb_pct'].rolling(40).mean()) / (feat['bb_pct'].rolling(40).std() + 1e-9)

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_10'] = tr.rolling(10).mean()
    feat['atr_norm'] = feat['atr_10'] / close

    # --- MACD ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    macd = ema8 - ema21
    feat['macd_signal'] = macd - macd.ewm(span=5, adjust=False).mean()

    # --- Volume ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_10'] = volume / (volume.rolling(10).mean() + 1e-9)
    feat['volume_z_20'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features ---
    trend_5 = (close / close.shift(5) - 1)
    trend_10 = (close / close.shift(10) - 1)
    trend_20 = (close / close.shift(20) - 1)
    feat['trend_dir_5'] = np.sign(trend_5)
    feat['trend_dir_10'] = np.sign(trend_10)
    feat['trend_dir_20'] = np.sign(trend_20)
    feat['trend_strength_5'] = trend_5.abs() * 10000
    feat['trend_strength_10'] = trend_10.abs() * 10000
    feat['trend_strength_20'] = trend_20.abs() * 10000

    realized_vol_10 = log_ret.rolling(10).std()
    realized_vol_20 = log_ret.rolling(20).std()
    realized_vol_40 = log_ret.rolling(40).std()
    feat['vol_regime'] = realized_vol_20 / (realized_vol_20.rolling(120).mean() + 1e-9)
    feat['vol_regime_fast'] = realized_vol_10 / (realized_vol_20 + 1e-9)
    feat['vol_regime_trend'] = (realized_vol_10 - realized_vol_40) / (realized_vol_40 + 1e-9)
    feat['vol_rank_120'] = realized_vol_20.rolling(120).rank(pct=True)

    # --- Mean reversion features ---
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    feat['price_sma5_dev'] = (close - sma_5) / (sma_5 + 1e-9)
    feat['price_sma20_dev'] = (close - sma_20) / (sma_20 + 1e-9)
    feat['sma_cross'] = (sma_5 - sma_20) / (sma_20 + 1e-9)

    # --- Momentum × Volatility interactions ---
    ret_3 = close.pct_change(3)
    feat['momentum_vol_interaction'] = ret_3 * realized_vol_20
    feat['momentum_regime'] = ret_3 / (realized_vol_20 + 1e-9)
    feat['momentum_regime_20'] = ret_3 * feat['vol_regime']
    feat['momentum_regime_fast'] = ret_3 * feat['vol_regime_fast']
    feat['momentum_x_voltrend'] = ret_3 * feat['vol_regime_trend']

    # --- Volume × Price position interactions ---
    vol_ratio_5 = volume / (volume.rolling(5).mean() + 1e-9)
    vol_ratio_10 = volume / (volume.rolling(10).mean() + 1e-9)
    bb_pct = (close - close.rolling(20).mean()) / (2 * close.rolling(20).std() + 1e-9)
    feat['volume_bb_interaction'] = vol_ratio_10 * bb_pct
    feat['volume_price_pressure'] = vol_ratio_5 * feat['price_sma5_dev']

    # --- High-low range features ---
    feat['hl_range'] = (high - low) / close
    feat['hl_range_ma'] = feat['hl_range'].rolling(5).mean()
    feat['candle_body_ratio'] = (close - df['open']).abs() / ((high - low) + 1e-9)
    feat['upper_wick_ratio'] = (high - df[['open', 'close']].max(axis=1)) / ((high - low) + 1e-9)
    feat['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - low) / ((high - low) + 1e-9)

    # --- Close position in range ---
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['close_position_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)
    feat['ret_z_20'] = (close.pct_change(1) - close.pct_change(1).rolling(20).mean()) / (close.pct_change(1).rolling(20).std() + 1e-9)
    feat['ret_accel_3'] = close.pct_change(1).diff(3)
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
    feat['efficiency_10'] = (close - close.shift(10)).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    feat['efficiency_20'] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20).sum() + 1e-9)
    feat['efficiency_trend'] = feat['efficiency_10'] - feat['efficiency_20']

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
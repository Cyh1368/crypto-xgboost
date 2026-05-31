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
    "n_estimators": 150,           # Fewer trees per regime model
    "max_depth": 3,                # Shallower to avoid overfitting
    "min_child_weight": 5,         # Higher to force robust patterns
    "reg_alpha": 0.1,              # Light L1 regularization
    "reg_lambda": 1.0,             # Moderate L2 regularization
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,         # Slower learning
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 20,
}

CLIP_PERCENTILE = 97
MIN_PRED_STD_RATIO = 0.15
N_REGIMES = 4                      # Number of market regimes
REGIME_WINDOW = 20                 # Window for regime features
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features used ONLY for regime detection.
    These capture market character but not predictive signals.
    """
    regime_feat = pd.DataFrame(index=df.index)
    
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Volatility regime
    log_ret = np.log(close / close.shift(1))
    regime_feat['vol_5'] = log_ret.rolling(5).std()
    regime_feat['vol_20'] = log_ret.rolling(20).std()
    
    # Trend regime
    regime_feat['trend_5'] = (close / close.shift(5) - 1)
    regime_feat['trend_20'] = (close / close.shift(20) - 1)
    
    # Volume regime
    regime_feat['volume_ratio'] = volume / (volume.rolling(20).mean() + 1e-9)
    
    # Range regime
    regime_feat['atr_ratio'] = (high - low) / close
    
    # Autocorrelation (momentum persistence)
    regime_feat['ret_autocorr'] = log_ret.rolling(20).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
    )
    
    return regime_feat.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with SHORT lookbacks and rolling z-score normalization.
    All features use max 20-bar lookback for regime robustness.
    """
    feat = pd.DataFrame(index=df.index)
    
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    # --- Short-term returns (1-20 bars max) ---
    for lag in [1, 2, 3, 5, 10, 20]:
        ret = close.pct_change(lag)
        # Z-score normalize within rolling window
        ret_mean = ret.rolling(20, min_periods=5).mean()
        ret_std = ret.rolling(20, min_periods=5).std()
        feat[f'ret_{lag}_z'] = (ret - ret_mean) / (ret_std + 1e-9)
    
    # --- Volatility features (short-term only) ---
    log_ret = np.log(close / close.shift(1))
    for w in [5, 10, 20]:
        vol = log_ret.rolling(w).std()
        vol_mean = vol.rolling(20, min_periods=5).mean()
        vol_std = vol.rolling(20, min_periods=5).std()
        feat[f'vol_{w}_z'] = (vol - vol_mean) / (vol_std + 1e-9)
    
    # --- Volatility ratios (regime-invariant) ---
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    feat['vol_ratio_5_20'] = vol_5 / (vol_20 + 1e-9)
    
    # --- RSI (normalized) ---
    for period in [5, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}_norm'] = (rsi - 50) / 50  # Center at 0
    
    # --- Bollinger position (short-term) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_position'] = (close - bb_mid) / (bb_std + 1e-9)
    
    # --- ATR normalized ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    feat['atr_norm'] = atr / close
    
    # --- MACD (short-term) ---
    ema6 = close.ewm(span=6, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    macd = ema6 - ema12
    macd_signal = macd.ewm(span=5, adjust=False).mean()
    macd_diff = macd - macd_signal
    feat['macd_norm'] = macd_diff / close
    
    # --- Volume features ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    feat['volume_ratio_5'] = volume / (vol_ma_5 + 1e-9)
    feat['volume_ratio_20'] = volume / (vol_ma_20 + 1e-9)
    
    # Volume-weighted price deviation
    vwap = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / close
    
    # --- Price position in range ---
    low_10 = low.rolling(10).min()
    high_10 = high.rolling(10).max()
    feat['price_position_10'] = (close - low_10) / (high_10 - low_10 + 1e-9)
    
    low_20 = low.rolling(20).min()
    high_20 = high.rolling(20).max()
    feat['price_position_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)
    
    # --- Mean reversion (short-term) ---
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    feat['price_sma5_dev'] = (close - sma_5) / sma_5
    feat['price_sma10_dev'] = (close - sma_10) / sma_10
    feat['price_sma20_dev'] = (close - sma_20) / sma_20
    feat['sma_cross_5_20'] = (sma_5 - sma_20) / sma_20
    
    # --- Momentum × Volatility (multiplicative interaction) ---
    ret_3 = close.pct_change(3)
    ret_5 = close.pct_change(5)
    vol_10 = log_ret.rolling(10).std()
    feat['momentum_vol_3'] = ret_3 * vol_10
    feat['momentum_vol_5'] = ret_5 * vol_10
    feat['risk_adj_momentum_3'] = ret_3 / (vol_10 + 1e-9)
    feat['risk_adj_momentum_5'] = ret_5 / (vol_10 + 1e-9)
    
    # --- Candle patterns ---
    body = (close - open_price).abs()
    range_hl = high - low
    feat['body_ratio'] = body / (range_hl + 1e-9)
    feat['upper_shadow'] = (high - close.clip(lower=open_price)) / (range_hl + 1e-9)
    feat['lower_shadow'] = (close.clip(upper=open_price) - low) / (range_hl + 1e-9)
    
    # --- Efficiency ratio (trend strength) ---
    price_change = (close - close.shift(10)).abs()
    path_length = close.diff().abs().rolling(10).sum()
    feat['efficiency_10'] = price_change / (path_length + 1e-9)
    
    # --- Return acceleration ---
    ret_1 = close.pct_change(1)
    feat['ret_accel_3'] = ret_1.diff(3)
    feat['ret_accel_5'] = ret_1.diff(5)
    
    # --- Volatility-volume correlation ---
    feat['vol_volume_corr'] = log_ret.rolling(20).corr(volume.pct_change(1))
    
    # --- Time features ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)
    
    # --- Orderbook (if available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(min(len(bids), 10))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(min(len(asks), 10))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
    
    # --- Funding rate (if available) ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_ma_8h'] = fr.rolling(32).mean()
        feat['funding_momentum'] = fr.diff(4)
    
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(method='ffill').fillna(0)
    return feat


def get_regime_specific_features(regime_id: int) -> list:
    """
    Return feature subset optimized for each regime.
    Regime 0: Low vol, mean reversion
    Regime 1: High vol, momentum
    Regime 2: Trending
    Regime 3: Ranging/choppy
    """
    # Base features used by all regimes
    base_features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'is_weekend',
        'ret_1_z', 'ret_2_z', 'vol_5_z', 'vol_10_z',
    ]
    
    if regime_id == 0:  # Low vol, mean reversion
        return base_features + [
            'price_sma5_dev', 'price_sma10_dev', 'price_sma20_dev',
            'bb_position', 'rsi_5_norm', 'rsi_14_norm',
            'price_position_10', 'price_position_20',
            'sma_cross_5_20', 'efficiency_10',
        ]
    elif regime_id == 1:  # High vol, momentum
        return base_features + [
            'ret_3_z', 'ret_5_z', 'ret_10_z',
            'momentum_vol_3', 'momentum_vol_5',
            'vol_ratio_5_20', 'ret_accel_3', 'ret_accel_5',
            'atr_norm', 'body_ratio',
        ]
    elif regime_id == 2:  # Trending
        return base_features + [
            'ret_5_z', 'ret_10_z', 'ret_20_z',
            'risk_adj_momentum_3', 'risk_adj_momentum_5',
            'macd_norm', 'efficiency_10', 'sma_cross_5_20',
            'volume_ratio_5', 'volume_ratio_20',
        ]
    else:  # regime_id == 3, Ranging/choppy
        return base_features + [
            'bb_position', 'rsi_5_norm', 'rsi_14_norm',
            'price_position_10', 'price_position_20',
            'upper_shadow', 'lower_shadow', 'body_ratio',
            'vwap_dev', 'vol_volume_corr',
        ]
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
import pandas as pd
import numpy as np
import joblib
import os

FEATURE_NAMES = [
    'obi_tau1', 'obi_tau3', 'obi_tau5', 'obi_tau10', 'spread_bps', 'depth_ratio_5',
    'depth_ratio_10', 'mid_price_move', 'book_pressure_3', 'kyle_lambda_est',
    'ret_1', 'ret_3', 'ret_6', 'ret_12', 'ret_48', 'vol_5', 'vol_20', 'vol_60',
    'rsi_14', 'rsi_6', 'macd_signal', 'bb_pct', 'atr_14', 'momentum_bar',
    'wick_ratio_up', 'wick_ratio_down', 'volume_ratio_5', 'volume_ratio_20',
    'vwap_dev', 'autocorr_5', 'skew_20', 'kurt_20', 'realized_vol_ratio',
    'trend_strength', 'close_rank_48', 'gap_open', 'overnight_ret',
    'funding_rate', 'funding_8h_ma', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_asia_session', 'is_us_session', 'is_weekend', 'minutes_to_funding'
]

BINARY_FEATURES = {'is_asia_session', 'is_us_session', 'is_weekend'}

def calc_obi(bids, asks, tau: float) -> float:
    if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
        return 0.5
    
    bids = np.array(bids)
    asks = np.array(asks)
    
    if bids.ndim == 1 and len(bids) > 0 and isinstance(bids[0], (list, np.ndarray)):
        bids = np.stack(bids)
    if asks.ndim == 1 and len(asks) > 0 and isinstance(asks[0], (list, np.ndarray)):
        asks = np.stack(asks)
    
    K = min(len(bids), len(asks))
    levels = np.arange(1, K + 1)
    weights = np.exp(-levels / tau)
    
    # Check if we have 3 columns [price, vol, timestamp] or 2 [price, vol]
    vol_idx = 1
    
    bid_vols = bids[:K, vol_idx]
    ask_vols = asks[:K, vol_idx]
    
    bid_w = (bid_vols * weights).sum()
    ask_w = (ask_vols * weights).sum()
    
    denom = bid_w + ask_w
    return bid_w / denom if denom > 0 else 0.5

def build_features(df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # --- Orderbook Microstructure ---
    for tau in [1, 3, 5, 10]:
        out[f'obi_tau{tau}'] = df.apply(lambda r: calc_obi(r['bids'], r['asks'], tau), axis=1)

    def get_best(row, side):
        if row[side] is None or len(row[side]) == 0: return np.nan
        return row[side][0][0]

    best_bid = df.apply(lambda r: get_best(r, 'bids'), axis=1)
    best_ask = df.apply(lambda r: get_best(r, 'asks'), axis=1)
    mid = (best_bid + best_ask) / 2
    
    out['spread_bps'] = (best_ask - best_bid) / mid * 10000
    
    def sum_vol(row, side, k_start, k_end):
        if row[side] is None: return 0.0
        return sum(item[1] for item in row[side][k_start:k_end])

    out['depth_ratio_5'] = df.apply(lambda r: sum_vol(r, 'bids', 0, 5), axis=1) / \
                           df.apply(lambda r: sum_vol(r, 'asks', 0, 5), axis=1).replace(0, np.nan)
    out['depth_ratio_10'] = df.apply(lambda r: sum_vol(r, 'bids', 0, 10), axis=1) / \
                            df.apply(lambda r: sum_vol(r, 'asks', 0, 10), axis=1).replace(0, np.nan)
    out['mid_price_move'] = mid.pct_change().fillna(0)
    out['book_pressure_3'] = df.apply(lambda r: sum_vol(r, 'bids', 1, 4), axis=1) / \
                             df.apply(lambda r: sum_vol(r, 'asks', 1, 4), axis=1).replace(0, np.nan)
    
    dp = df['close'].diff().abs()
    dv = df['volume'].diff().abs().replace(0, np.nan)
    out['kyle_lambda_est'] = (dp / dv).rolling(20).mean().fillna(0)

    # --- Price Action ---
    log_ret = np.log(df['close'] / df['close'].shift(1))
    for lag in [1, 3, 6, 12, 48]:
        out[f'ret_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
    
    out['vol_5'] = log_ret.rolling(5).std()
    out['vol_20'] = log_ret.rolling(20).std()
    out['vol_60'] = log_ret.rolling(60).std()
    
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50) / 100

    out['rsi_14'] = calc_rsi(df['close'], 14)
    out['rsi_6'] = calc_rsi(df['close'], 6)
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out['macd_signal'] = macd.ewm(span=9, adjust=False).mean()
    
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    out['bb_pct'] = ((df['close'] - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)).clip(0, 1)
    
    out['atr_14'] = (df['high'] - df['low']).rolling(14).mean() / df['close']
    
    bar_range = (df['high'] - df['low']).replace(0, 1e-9)
    out['momentum_bar'] = (df['close'] - df['open']) / bar_range
    out['wick_ratio_up'] = (df['high'] - df[['open', 'close']].max(axis=1)) / bar_range
    out['wick_ratio_down'] = (df[['open', 'close']].min(axis=1) - df['low']) / bar_range
    
    out['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean().replace(0, np.nan)
    out['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)
    
    vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    out['vwap_dev'] = (df['close'] - vwap) / (vwap + 1e-9)
    
    out['autocorr_5'] = log_ret.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
    out['skew_20'] = log_ret.rolling(20).skew()
    out['kurt_20'] = log_ret.rolling(20).kurt()
    out['realized_vol_ratio'] = out['vol_5'] / out['vol_60'].replace(0, np.nan)
    out['trend_strength'] = out['ret_12'].abs() / out['vol_20'].replace(0, np.nan)
    
    out['close_rank_48'] = df['close'].rolling(48).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    out['gap_open'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    out['overnight_ret'] = out['gap_open']

    # --- Macro ---
    out['funding_rate'] = df['funding_rate']
    out['funding_8h_ma'] = df['funding_rate'].rolling(8).mean()

    # --- Time ---
    hour = df.index.hour
    dow = df.index.dayofweek
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    out['is_asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    out['is_us_session'] = ((hour >= 13) & (hour < 21)).astype(int)
    out['is_weekend'] = (dow >= 5).astype(int)
    
    def mins_to_next(ts):
        next_funding_hour = ((ts.hour // 8) + 1) * 8
        if next_funding_hour == 24:
            next_ts = ts.normalize() + pd.Timedelta(days=1)
        else:
            next_ts = ts.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)
        return (next_ts - ts).total_seconds() / 60
    out['minutes_to_funding'] = pd.Series(df.index).apply(mins_to_next).values

    out = out.dropna()
    out = out[FEATURE_NAMES]

    if scaler is not None:
        out = pd.DataFrame(scaler.transform(out), index=out.index, columns=out.columns)

    return out

import numpy as np
import pandas as pd
from .registry import registry

TAU_VALUES = [1, 3, 5, 10]

def obi(bids, asks, tau: float) -> float:
    """
    Computes exponentially-decayed weighted imbalance.
    bids: shape (K, 2) — [[price, vol], ...]  sorted best→worst
    asks: shape (K, 2) — [[price, vol], ...]  sorted best→worst
    Returns OBI ∈ [0, 1].
    """
    if bids is None or asks is None:
        return 0.5
        
    bids = np.array(bids)
    asks = np.array(asks)
    
    if bids.ndim == 1 and len(bids) > 0 and isinstance(bids[0], (list, np.ndarray)):
        # Handle case where numpy makes a 1D array of objects
        bids = np.stack(bids)
        asks = np.stack(asks)
    
    if len(bids) == 0 or len(asks) == 0:
        return 0.5
        
    K = min(len(bids), len(asks))
    levels = np.arange(1, K + 1)
    weights = np.exp(-levels / tau)
    
    bid_vols = bids[:K, 1]
    ask_vols = asks[:K, 1]
    
    bid_w = (bid_vols * weights).sum()
    ask_w = (ask_vols * weights).sum()
    
    denom = bid_w + ask_w
    return float(bid_w / denom) if denom > 0 else 0.5

@registry.register("obi_tau1")
def feature_obi_tau1(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: obi(row['bids'], row['asks'], 1), axis=1)

@registry.register("obi_tau3")
def feature_obi_tau3(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: obi(row['bids'], row['asks'], 3), axis=1)

@registry.register("obi_tau5")
def feature_obi_tau5(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: obi(row['bids'], row['asks'], 5), axis=1)

@registry.register("obi_tau10")
def feature_obi_tau10(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: obi(row['bids'], row['asks'], 10), axis=1)

@registry.register("spread_bps")
def feature_spread_bps(df: pd.DataFrame) -> pd.Series:
    def calc_spread(row):
        if row['bids'] is None or row['asks'] is None or len(row['bids']) == 0 or len(row['asks']) == 0:
            return 0.0
        best_bid = row['bids'][0][0]
        best_ask = row['asks'][0][0]
        mid = (best_bid + best_ask) / 2
        return (best_ask - best_bid) / mid * 10000
    return df.apply(calc_spread, axis=1)

@registry.register("depth_ratio_5")
def feature_depth_ratio_5(df: pd.DataFrame) -> pd.Series:
    def calc_depth_ratio(row, k=5):
        if row['bids'] is None or row['asks'] is None: return 1.0
        b_vol = sum(b[1] for b in row['bids'][:k])
        a_vol = sum(a[1] for a in row['asks'][:k])
        return b_vol / a_vol if a_vol > 0 else 1.0
    return df.apply(lambda r: calc_depth_ratio(r, 5), axis=1)

@registry.register("depth_ratio_10")
def feature_depth_ratio_10(df: pd.DataFrame) -> pd.Series:
    def calc_depth_ratio(row, k=10):
        if row['bids'] is None or row['asks'] is None: return 1.0
        b_vol = sum(b[1] for b in row['bids'][:k])
        a_vol = sum(a[1] for a in row['asks'][:k])
        return b_vol / a_vol if a_vol > 0 else 1.0
    return df.apply(lambda r: calc_depth_ratio(r, 10), axis=1)

@registry.register("mid_price_move")
def feature_mid_price_move(df: pd.DataFrame) -> pd.Series:
    def get_mid(row):
        if row['bids'] is None or row['asks'] is None or len(row['bids']) == 0 or len(row['asks']) == 0:
            return np.nan
        return (row['bids'][0][0] + row['asks'][0][0]) / 2
    mids = df.apply(get_mid, axis=1)
    return mids.pct_change().fillna(0)

@registry.register("book_pressure_3")
def feature_book_pressure_3(df: pd.DataFrame) -> pd.Series:
    def calc_pressure(row):
        if row['bids'] is None or row['asks'] is None: return 1.0
        b_vol = sum(b[1] for b in row['bids'][1:4])
        a_vol = sum(a[1] for a in row['asks'][1:4])
        return b_vol / a_vol if a_vol > 0 else 1.0
    return df.apply(calc_pressure, axis=1)

@registry.register("vol_at_spread")
def feature_vol_at_spread(df: pd.DataFrame) -> pd.Series:
    def calc_vol(row):
        if row['bids'] is None or row['asks'] is None or len(row['bids']) == 0 or len(row['asks']) == 0:
            return 0.0
        vol = row['bids'][0][1] + row['asks'][0][1]
        return vol
    vols = df.apply(calc_vol, axis=1)
    # Normalize by rolling 1-hour average (4 bars of 15-min)
    return vols / vols.rolling(4).mean().fillna(vols)

@registry.register("kyle_lambda_est")
def feature_kyle_lambda_est(df: pd.DataFrame) -> pd.Series:
    # ΔP / ΔV proxy
    mid_diff = feature_mid_price_move(df).abs()
    vol = df['volume']
    return (mid_diff / (vol + 1e-9)).fillna(0)

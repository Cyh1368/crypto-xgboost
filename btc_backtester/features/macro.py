import numpy as np
import pandas as pd
from .registry import registry

@registry.register("funding_rate")
def feature_funding_rate(df: pd.DataFrame) -> pd.Series:
    return df['funding_rate'] if 'funding_rate' in df.columns else pd.Series(0, index=df.index)

@registry.register("funding_8h_ma")
def feature_funding_8h_ma(df: pd.DataFrame) -> pd.Series:
    return feature_funding_rate(df).rolling(8).mean()

@registry.register("oi_chg_1")
def feature_oi_chg_1(df: pd.DataFrame) -> pd.Series:
    return df['open_interest'].pct_change(1) if 'open_interest' in df.columns else pd.Series(0, index=df.index)

@registry.register("oi_chg_6")
def feature_oi_chg_6(df: pd.DataFrame) -> pd.Series:
    return df['open_interest'].pct_change(6) if 'open_interest' in df.columns else pd.Series(0, index=df.index)

@registry.register("oi_vol_ratio")
def feature_oi_vol_ratio(df: pd.DataFrame) -> pd.Series:
    if 'open_interest' in df.columns and 'volume' in df.columns:
        vol_24h = df['volume'].rolling(96).sum() # 96 * 15m = 24h
        return df['open_interest'] / (vol_24h + 1e-9)
    return pd.Series(0, index=df.index)

@registry.register("basis_pct")
def feature_basis_pct(df: pd.DataFrame) -> pd.Series:
    if 'futures_price' in df.columns and 'spot_price' in df.columns:
        return (df['futures_price'] - df['spot_price']) / df['spot_price']
    return pd.Series(0, index=df.index)

@registry.register("liquidation_buy_1h")
def feature_liquidation_buy_1h(df: pd.DataFrame) -> pd.Series:
    return df['liq_buy'].rolling(4).sum() if 'liq_buy' in df.columns else pd.Series(0, index=df.index)

@registry.register("liquidation_sell_1h")
def feature_liquidation_sell_1h(df: pd.DataFrame) -> pd.Series:
    return df['liq_sell'].rolling(4).sum() if 'liq_sell' in df.columns else pd.Series(0, index=df.index)

@registry.register("liq_imbalance")
def feature_liq_imbalance(df: pd.DataFrame) -> pd.Series:
    buy = feature_liquidation_buy_1h(df)
    sell = feature_liquidation_sell_1h(df)
    return (buy - sell) / (buy + sell + 1e-9)

@registry.register("fear_greed_idx")
def feature_fear_greed_idx(df: pd.DataFrame) -> pd.Series:
    return df['fear_greed'].ffill() if 'fear_greed' in df.columns else pd.Series(50, index=df.index)

@registry.register("btc_dominance_chg")
def feature_btc_dominance_chg(df: pd.DataFrame) -> pd.Series:
    return df['btc_dominance'].diff() if 'btc_dominance' in df.columns else pd.Series(0, index=df.index)

@registry.register("hour_sin")
def feature_hour_sin(df: pd.DataFrame) -> pd.Series:
    return np.sin(2 * np.pi * df.index.hour / 24)

@registry.register("hour_cos")
def feature_hour_cos(df: pd.DataFrame) -> pd.Series:
    return np.cos(2 * np.pi * df.index.hour / 24)

@registry.register("dow_sin")
def feature_dow_sin(df: pd.DataFrame) -> pd.Series:
    return np.sin(2 * np.pi * df.index.dayofweek / 7)

@registry.register("dow_cos")
def feature_dow_cos(df: pd.DataFrame) -> pd.Series:
    return np.cos(2 * np.pi * df.index.dayofweek / 7)

@registry.register("is_asia_session")
def feature_is_asia_session(df: pd.DataFrame) -> pd.Series:
    return ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)

@registry.register("is_us_session")
def feature_is_us_session(df: pd.DataFrame) -> pd.Series:
    return ((df.index.hour >= 13) & (df.index.hour < 20)).astype(int)

@registry.register("is_weekend")
def feature_is_weekend(df: pd.DataFrame) -> pd.Series:
    return (df.index.dayofweek >= 5).astype(int)

@registry.register("minutes_to_funding")
def feature_minutes_to_funding(df: pd.DataFrame) -> pd.Series:
    # Funding usually at 0, 8, 16 UTC
    def mins_to_next(ts):
        next_funding_hour = ((ts.hour // 8) + 1) * 8
        if next_funding_hour == 24:
            next_ts = ts.normalize() + pd.Timedelta(days=1)
        else:
            next_ts = ts.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)
        return (next_ts - ts).total_seconds() / 60
    return pd.Series(df.index).apply(mins_to_next).values

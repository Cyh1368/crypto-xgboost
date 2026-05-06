import numpy as np
import pandas as pd
from .registry import registry

def log_return(series: pd.Series, periods: int) -> pd.Series:
    return np.log(series / series.shift(periods))

@registry.register("ret_1")
def feature_ret_1(df: pd.DataFrame) -> pd.Series: return log_return(df['close'], 1)

@registry.register("ret_3")
def feature_ret_3(df: pd.DataFrame) -> pd.Series: return log_return(df['close'], 3)

@registry.register("ret_6")
def feature_ret_6(df: pd.DataFrame) -> pd.Series: return log_return(df['close'], 6)

@registry.register("ret_12")
def feature_ret_12(df: pd.DataFrame) -> pd.Series: return log_return(df['close'], 12)

@registry.register("ret_48")
def feature_ret_48(df: pd.DataFrame) -> pd.Series: return log_return(df['close'], 48)

@registry.register("vol_5")
def feature_vol_5(df: pd.DataFrame) -> pd.Series: return feature_ret_1(df).rolling(5).std()

@registry.register("vol_20")
def feature_vol_20(df: pd.DataFrame) -> pd.Series: return feature_ret_1(df).rolling(20).std()

@registry.register("vol_60")
def feature_vol_60(df: pd.DataFrame) -> pd.Series: return feature_ret_1(df).rolling(60).std()

@registry.register("rsi_14")
def feature_rsi_14(df: pd.DataFrame) -> pd.Series:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@registry.register("rsi_6")
def feature_rsi_6(df: pd.DataFrame) -> pd.Series:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@registry.register("macd_signal")
def feature_macd_signal(df: pd.DataFrame) -> pd.Series:
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

@registry.register("bb_pct")
def feature_bb_pct(df: pd.DataFrame) -> pd.Series:
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    return (df['close'] - lower) / (upper - lower + 1e-9)

@registry.register("atr_14")
def feature_atr_14(df: pd.DataFrame) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(14).mean() / df['close']

@registry.register("momentum_bar")
def feature_momentum_bar(df: pd.DataFrame) -> pd.Series:
    return (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)

@registry.register("wick_ratio_up")
def feature_wick_ratio_up(df: pd.DataFrame) -> pd.Series:
    atr = feature_atr_14(df) * df['close']
    return (df['high'] - df[['open', 'close']].max(axis=1)) / (atr + 1e-9)

@registry.register("wick_ratio_down")
def feature_wick_ratio_down(df: pd.DataFrame) -> pd.Series:
    atr = feature_atr_14(df) * df['close']
    return (df[['open', 'close']].min(axis=1) - df['low']) / (atr + 1e-9)

@registry.register("volume_ratio_5")
def feature_volume_ratio_5(df: pd.DataFrame) -> pd.Series:
    return df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)

@registry.register("volume_ratio_20")
def feature_volume_ratio_20(df: pd.DataFrame) -> pd.Series:
    return df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)

@registry.register("vwap_dev")
def feature_vwap_dev(df: pd.DataFrame) -> pd.Series:
    vwap = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
    return (df['close'] - vwap) / df['close']

@registry.register("autocorr_5")
def feature_autocorr_5(df: pd.DataFrame) -> pd.Series:
    ret = feature_ret_1(df)
    return ret.rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0)

@registry.register("skew_20")
def feature_skew_20(df: pd.DataFrame) -> pd.Series:
    return feature_ret_1(df).rolling(20).skew()

@registry.register("kurt_20")
def feature_kurt_20(df: pd.DataFrame) -> pd.Series:
    return feature_ret_1(df).rolling(20).kurt()

@registry.register("realized_vol_ratio")
def feature_realized_vol_ratio(df: pd.DataFrame) -> pd.Series:
    return feature_vol_5(df) / (feature_vol_60(df) + 1e-9)

@registry.register("trend_strength")
def feature_trend_strength(df: pd.DataFrame) -> pd.Series:
    # ADX(14) simplified
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    
    tr = feature_atr_14(df) * df['close'] # Approximate TR
    plus_di = 100 * (plus_dm.rolling(14).mean() / (tr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (tr + 1e-9))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return dx.rolling(14).mean()

@registry.register("close_rank_48")
def feature_close_rank_48(df: pd.DataFrame) -> pd.Series:
    return df['close'].rolling(48).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

@registry.register("gap_open")
def feature_gap_open(df: pd.DataFrame) -> pd.Series:
    return (df['open'] - df['close'].shift()) / (df['close'].shift() + 1e-9)

@registry.register("overnight_ret")
def feature_overnight_ret(df: pd.DataFrame) -> pd.Series:
    # return during exchange low-volume window (22:00 to 02:00 UTC)
    def is_overnight(ts):
        return ts.hour >= 22 or ts.hour < 2
    overnight_mask = pd.Series(df.index).apply(is_overnight).values
    ret = feature_ret_1(df)
    return ret.where(overnight_mask, 0)

import pandas as pd
import numpy as np

def calculate_indicators(df):
    """
    Calculates technical indicators as specified in the research paper.
    """
    # Copy to avoid modifying original
    df = df.copy()
    
    # EMA
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['RSI10'] = calculate_rsi(df['close'], 10)
    df['RSI14'] = calculate_rsi(df['close'], 14)
    df['RSI30'] = calculate_rsi(df['close'], 30)
    df['RSI200'] = calculate_rsi(df['close'], 200)
    
    # MOM (Momentum)
    df['MOM10'] = df['close'].diff(10)
    df['MOM30'] = df['close'].diff(30)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # %K (Stochastic Oscillator)
    def calculate_stoch_k(df, period):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        return 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)

    df['%K10'] = calculate_stoch_k(df, 10)
    df['%K30'] = calculate_stoch_k(df, 30)
    df['%K200'] = calculate_stoch_k(df, 200)
    
    # PROC (Price Rate of Change)
    df['PROC9'] = df['close'].pct_change(9) * 100
    
    return df

def prepare_features(df):
    """
    Prepares the full feature set including historical and technical indicators.
    """
    df = calculate_indicators(df)
    
    # Ensure we have the basic historical columns if they exist
    # If QAV, NOT, TBBV, TBQV are missing, we just skip them but keep placeholders or log warning
    paper_cols = [
        'close', 'volume', 'QAV', 'NOT', 'TBBV', 'TBQV',
        'RSI14', 'RSI30', 'RSI200', 'MOM10', 'MOM30', 'MACD', 'PROC9',
        'EMA10', 'EMA30', 'EMA200', '%K10', '%K30', '%K200'
    ]
    
    # Check which ones are available
    available_cols = [c for c in paper_cols if c in df.columns]
    
    return df.dropna()

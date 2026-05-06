import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from typing import Optional

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Loads OHLCV + L2 orderbook data from a Parquet file.
        Expected schema:
        - timestamp: datetime
        - open, high, low, close, volume: float
        - bids: list of [price, vol] (K levels)
        - asks: list of [price, vol] (K levels)
        """
        df = pd.read_parquet(self.file_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        return df

def get_snapshot(df: pd.DataFrame, timestamp, side='bids') -> np.ndarray:
    """Helper to extract bids/asks as numpy array at a specific timestamp."""
    val = df.loc[timestamp, side]
    if isinstance(val, list):
        return np.array(val)
    return val

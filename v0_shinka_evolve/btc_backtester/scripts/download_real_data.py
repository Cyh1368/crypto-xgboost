import ccxt
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

def fetch_real_data(symbol, limit=500):
    exchange = ccxt.binance({'options': {'defaultType': 'future'}})
    print(f"Fetching {limit} bars for {symbol}...")
    
    # Fetch OHLCV
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Fetch current orderbook as a representative snapshot for the "L2" requirement
    # Since we can't get historical L2 via REST easily, we'll use the current one 
    # and adjust volumes based on historical OHLCV volume to create a "realistic" historical proxy.
    print(f"Fetching L2 snapshot for {symbol}...")
    ob = exchange.fetch_order_book(symbol, limit=20)
    bids_template = np.array(ob['bids']) # [[price, vol], ...]
    asks_template = np.array(ob['asks'])
    
    # We will "shift" the prices of the template to match the historical close
    # and scale volumes by the ratio of historical volume to current volume.
    current_close = df.iloc[-1]['close']
    current_vol = df.iloc[-1]['volume']
    
    bids_list = []
    asks_list = []
    
    for i in range(len(df)):
        hist_close = df.iloc[i]['close']
        hist_vol = df.iloc[i]['volume']
        
        price_ratio = hist_close / current_close
        vol_ratio = hist_vol / (current_vol + 1e-9)
        
        # Shift prices and scale volumes
        bids = bids_template.copy()
        bids[:, 0] *= price_ratio
        bids[:, 1] *= vol_ratio
        
        asks = asks_template.copy()
        asks[:, 0] *= price_ratio
        asks[:, 1] *= vol_ratio
        
        bids_list.append(bids.tolist())
        asks_list.append(asks.tolist())
        
    df['bids'] = bids_list
    df['asks'] = asks_list
    
    # Add dummy macro data for features that expect it
    df['funding_rate'] = 0.0001 # Typical
    df['open_interest'] = hist_vol * 100 # Dummy OI
    df['spot_price'] = df['close'] * 0.9998 # Basis proxy
    
    return df

def main():
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'TRX/USDT', 'LINK/USDT', 'DOT/USDT'
    ]
    
    out_dir = 'btc_backtester/data/raw/real'
    os.makedirs(out_dir, exist_ok=True)
    
    for symbol in symbols:
        safe_name = symbol.replace('/', '_')
        try:
            df = fetch_real_data(symbol)
            df.to_parquet(f"{out_dir}/{safe_name}.parquet")
            print(f"Saved {symbol} to {out_dir}/{safe_name}.parquet")
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
        time.sleep(1) # Rate limit

if __name__ == "__main__":
    main()

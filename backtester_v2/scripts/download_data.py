import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timezone, timedelta

def fetch_coinbase_candles(start_ts, end_ts, granularity='FIFTEEN_MINUTE'):
    """Fetch Coinbase BTC-USD candles."""
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles"
    params = {
        "start": str(int(start_ts)),
        "end": str(int(end_ts)),
        "granularity": granularity
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return []
    return resp.json().get('candles', [])

def download_btc_data(limit=50000):
    print(f"Downloading {limit} BTC/USD candles from Coinbase...")
    
    all_candles = []
    # 15 minute granularity
    granularity_seconds = 15 * 60
    # Coinbase limit per call is 350 candles
    chunk_size = 300 
    
    end_dt = datetime.now(timezone.utc)
    current_end = int(end_dt.timestamp())
    
    while len(all_candles) < limit:
        current_start = current_end - (chunk_size * granularity_seconds)
        print(f"  Fetching {chunk_size} candles ending at {datetime.fromtimestamp(current_end, tz=timezone.utc)}...")
        
        candles = fetch_coinbase_candles(current_start, current_end)
        if not candles:
            print("  No more candles found or error occurred.")
            break
            
        all_candles.extend(candles)
        # Sort by start timestamp to ensure we don't have overlaps or gaps if possible
        # Actually Coinbase returns them in some order, usually desc
        
        # Next chunk end should be the start of the earliest candle fetched minus 1
        earliest_start = min(int(c['start']) for c in candles)
        current_end = earliest_start - 1
        
        print(f"  Total candles so far: {len(all_candles)}")
        time.sleep(0.2) # Respect rate limits
        
    df = pd.DataFrame(all_candles)
    df['start'] = df['start'].astype(int)
    df = df.drop_duplicates(subset=['start']).sort_values('start')
    
    # Trim to exactly limit if we over-fetched
    df = df.tail(limit)
    
    df['timestamp'] = pd.to_datetime(df['start'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    # Ensure float types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    # Proxy Orderbook (since historical L2 is not available for free)
    print("Proxying order book and funding rate...")
    # Fetch a live snapshot for template
    try:
        # We can use Kraken for a free snapshot if Coinbase advanced needs auth for OB
        # Or just use a hardcoded template if we want to be fast, 
        # but let's try Kraken for a better proxy.
        import ccxt
        kraken = ccxt.kraken()
        ob = kraken.fetch_order_book('BTC/USDT', limit=20)
        bids_template = np.array(ob['bids'])
        asks_template = np.array(ob['asks'])
    except Exception as e:
        print(f"  Could not fetch live OB: {e}. Using synthetic template.")
        # Synthetic template: 20 levels, 0.5 bps spread each, decaying volume
        bids_template = np.array([[1.0 - i*0.0001, 1.0 * np.exp(-i/5)] for i in range(1, 21)])
        asks_template = np.array([[1.0 + i*0.0001, 1.0 * np.exp(-i/5)] for i in range(1, 21)])

    current_close = df.iloc[-1]['close']
    current_vol = df.iloc[-1]['volume']
    
    bids_list = []
    asks_list = []
    
    # We will compute the features directly to save space or just store lists
    # Storing lists in Parquet is fine.
    for i in range(len(df)):
        hist_close = df.iloc[i]['close']
        hist_vol = df.iloc[i]['volume']
        
        price_ratio = hist_close / (current_close if current_close != 0 else 1.0)
        vol_ratio = hist_vol / (current_vol if current_vol != 0 else 1.0)
        
        b = bids_template.copy()
        b[:, 0] *= price_ratio
        b[:, 1] *= vol_ratio
        
        a = asks_template.copy()
        a[:, 0] *= price_ratio
        a[:, 1] *= vol_ratio
        
        bids_list.append(b.tolist())
        asks_list.append(a.tolist())
        
    df['bids'] = bids_list
    df['asks'] = asks_list
    df['funding_rate'] = 0.0001 # Proxy constant
    
    return df

if __name__ == "__main__":
    out_dir = 'backtester_v2/data/raw'
    os.makedirs(out_dir, exist_ok=True)
    
    df = download_btc_data(limit=50000)
    if df is not None:
        df.to_parquet(f"{out_dir}/BTC_USDT_50k.parquet")
        print(f"Saved to {out_dir}/BTC_USDT_50k.parquet")

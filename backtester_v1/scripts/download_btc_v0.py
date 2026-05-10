import ccxt
import pandas as pd
import numpy as np
import os
import time

def fetch_btc_data(symbol='BTC/USDT', limit=50000):
    exchange = ccxt.binanceus()
    print(f"Fetching {limit} bars for {symbol} from Binance US...")
    
    all_ohlcv = []
    # Binance US fetch_ohlcv limit is usually 1000.
    current_since = exchange.milliseconds() - (limit * 15 * 60 * 1000)
    
    while len(all_ohlcv) < limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', since=current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            if len(all_ohlcv) % 5000 == 0 or len(all_ohlcv) == limit:
                print(f"  Fetched {len(all_ohlcv)} bars so far...")
            time.sleep(0.1)
        except Exception as e:
            print(f"  Error fetching: {e}")
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Proxying order book for {symbol}...")
    ob = exchange.fetch_order_book(symbol, limit=20)
    bids_template = np.array(ob['bids'])
    asks_template = np.array(ob['asks'])
    
    current_close = df.iloc[-1]['close']
    current_vol = df.iloc[-1]['volume']
    
    bids_list = []
    asks_list = []
    
    # Efficiently create proxy book
    for i in range(len(df)):
        hist_close = df.iloc[i]['close']
        hist_vol = df.iloc[i]['volume']
        
        price_ratio = hist_close / (current_close + 1e-9)
        vol_ratio = hist_vol / (current_vol + 1e-9)
        
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
    df['funding_rate'] = 0.0001 # Proxy
    
    return df

def main():
    symbol = 'BTC/USDT'
    out_dir = 'backtester_v1/data/raw'
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        df = fetch_btc_data(symbol, limit=50000)
        safe_name = 'BTC_USDT_v0'
        df.to_parquet(f"{out_dir}/{safe_name}.parquet")
        print(f"Saved {symbol} to {out_dir}/{safe_name}.parquet")
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")

if __name__ == "__main__":
    main()

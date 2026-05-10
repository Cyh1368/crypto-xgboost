import ccxt
import pandas as pd
import numpy as np
import os
import time

def fetch_futures_data(symbol, limit=5000):
    exchange = ccxt.kucoin()
    print(f"Fetching {limit} bars for {symbol}...")
    
    all_ohlcv = []
    # KuCoin limit per call is 1500
    current_since = exchange.milliseconds() - (limit * 15 * 60 * 1000)
    
    while len(all_ohlcv) < limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', since=current_since, limit=1500)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            print(f"  Fetched {len(all_ohlcv)} bars so far...")
            time.sleep(1)
        except Exception as e:
            print(f"  Error fetching: {e}")
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetching L2 snapshot for {symbol}...")
    ob = exchange.fetch_order_book(symbol, limit=20)
    bids_template = np.array(ob['bids'])
    asks_template = np.array(ob['asks'])
    
    current_close = df.iloc[-1]['close']
    current_vol = df.iloc[-1]['volume']
    
    bids_list = []
    asks_list = []
    
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
    df['open_interest'] = df['volume'] * 100 # Proxy
    
    return df

def main():
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'LINK/USDT', 'LTC/USDT'
    ]
    
    out_dir = 'backtester_v1/data/raw/multi'
    os.makedirs(out_dir, exist_ok=True)
    
    for symbol in symbols:
        safe_name = symbol.replace('/', '_')
        try:
            df = fetch_futures_data(symbol)
            df.to_parquet(f"{out_dir}/{safe_name}.parquet")
            print(f"Saved {symbol} to {out_dir}/{safe_name}.parquet")
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()

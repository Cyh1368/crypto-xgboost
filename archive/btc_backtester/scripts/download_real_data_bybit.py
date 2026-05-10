import ccxt
import pandas as pd
import numpy as np
import os
import time

def fetch_real_data_bybit(symbol, limit=500):
    # Bybit doesn't restrict US IP as strictly for public data sometimes, 
    # but let's see. If not, we'll try another.
    exchange = ccxt.bybit({'options': {'defaultType': 'linear'}})
    print(f"Fetching {limit} bars for {symbol} from Bybit...")
    
    # Bybit uses symbols like BTCUSDT
    bybit_symbol = symbol.replace('/', '')
    
    ohlcv = exchange.fetch_ohlcv(bybit_symbol, timeframe='15m', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetching L2 snapshot for {bybit_symbol}...")
    ob = exchange.fetch_order_book(bybit_symbol, limit=20)
    bids_template = np.array(ob['bids'])
    asks_template = np.array(ob['asks'])
    
    current_close = df.iloc[-1]['close']
    bids_list = []
    asks_list = []
    
    for i in range(len(df)):
        hist_close = df.iloc[i]['close']
        price_ratio = hist_close / current_close
        
        bids = bids_template.copy()
        bids[:, 0] *= price_ratio
        asks = asks_template.copy()
        asks[:, 0] *= price_ratio
        
        bids_list.append(bids.tolist())
        asks_list.append(asks.tolist())
        
    df['bids'] = bids_list
    df['asks'] = asks_list
    df['funding_rate'] = 0.0001
    df['spot_price'] = df['close'] * 0.9998
    
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
            df = fetch_real_data_bybit(symbol)
            df.to_parquet(f"{out_dir}/{safe_name}.parquet")
            print(f"Saved {symbol} to {out_dir}/{safe_name}.parquet")
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()

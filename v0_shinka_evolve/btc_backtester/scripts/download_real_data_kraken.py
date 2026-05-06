import ccxt
import pandas as pd
import numpy as np
import os
import time

def fetch_real_data_kraken(symbol, limit=500):
    exchange = ccxt.kraken()
    print(f"Fetching {limit} bars for {symbol} from Kraken...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetching L2 snapshot for {symbol}...")
    ob = exchange.fetch_order_book(symbol, limit=20)
    bids_template = np.array(ob['bids'])
    asks_template = np.array(ob['asks'])
    
    current_close = df.iloc[-1]['close']
    bids_list = []
    asks_list = []
    for i in range(len(df)):
        hist_close = df.iloc[i]['close']
        price_ratio = hist_close / (current_close + 1e-9)
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
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'XRP/USD',
        'DOGE/USD', 'DOT/USD', 'LTC/USD', 'LINK/USD', 'BCH/USD'
    ]
    out_dir = 'btc_backtester/data/raw/real'
    os.makedirs(out_dir, exist_ok=True)
    for symbol in symbols:
        safe_name = symbol.replace('/', '_')
        try:
            df = fetch_real_data_kraken(symbol)
            df.to_parquet(f"{out_dir}/{safe_name}.parquet")
            print(f"Saved {symbol} to {out_dir}/{safe_name}.parquet")
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()

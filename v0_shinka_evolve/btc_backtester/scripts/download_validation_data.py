import ccxt
import pandas as pd
import numpy as np
import time
import os

def download_btc_data(limit=5000):
    # Use Bybit as an alternative if Binance is restricted, or try a different Binance endpoint
    # Actually, let's try Kraken or another exchange that might be less restricted if Binance fails.
    # But first, try to fetch from Binance with a different approach or just use Kraken.
    exchange = ccxt.kraken()
    symbol = 'BTC/USDT'
    print(f"Downloading {limit} bars of {symbol} from {exchange.id}...")
    
    all_ohlcv = []
    # Kraken fetch_ohlcv doesn't support 'since' for '15m' as easily for deep history in one go
    # but we can try to loop if needed.
    # Kraken 15m limit is usually around 720 bars.
    
    current_since = exchange.milliseconds() - (limit * 15 * 60 * 1000)
    
    while len(all_ohlcv) < limit:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', since=current_since)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        current_since = ohlcv[-1][0] + 1
        print(f"Fetched {len(all_ohlcv)}/{limit}...")
        time.sleep(1)
    
    df = pd.DataFrame(all_ohlcv[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Generate noisy orderbook proxy since Kraken REST historical doesn't have it
    # This is consistent with our training data generation for "real" files
    ob = exchange.fetch_order_book(symbol, limit=20)
    b_temp = ob['bids']
    a_temp = ob['asks']
    
    bids_l = []
    asks_l = []
    for i in range(len(df)):
        r = df.iloc[i]['close'] / df.iloc[-1]['close']
        v = df.iloc[i]['volume'] / (df.iloc[-1]['volume'] + 1e-9)
        noise_b = np.random.uniform(0.8, 1.2, size=(len(b_temp),))
        noise_a = np.random.uniform(0.8, 1.2, size=(len(a_temp),))
        
        bids_l.append([[item[0]*r, item[1]*v*n] for item, n in zip(b_temp, noise_b)])
        asks_l.append([[item[0]*r, item[1]*v*n] for item, n in zip(a_temp, noise_a)])
        
    df['bids'] = bids_l
    df['asks'] = asks_l
    df['funding_rate'] = 0.0001
    df['open_interest'] = df['volume'] * 100
    df['spot_price'] = df['close'] # Proxy
    
    out_path = 'v0_shinka_evolve/btc_backtester/data/raw/btc_5000_validation.parquet'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved validation data to {out_path}")

if __name__ == "__main__":
    download_btc_data()

import requests
import pandas as pd
import numpy as np
import os
import time
import zipfile
import io
import ccxt
from datetime import datetime, timezone, timedelta

import sys

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

def fetch_kraken_futures_data(symbol='PI_XBTUSD', limit=50000):
    flush_print(f"Fetching {limit} bars of {symbol} OHLCV and Funding from Kraken Futures...")
    exchange = ccxt.krakenfutures()
    ohlcv_list = []
    current_since = exchange.milliseconds() - (limit * 15 * 60 * 1000)
    
    while len(ohlcv_list) < limit:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe='15m', since=current_since, limit=500)
            if not batch: break
            ohlcv_list.extend(batch)
            current_since = batch[-1][0] + 1
            if len(ohlcv_list) % 5000 == 0: flush_print(f"  Fetched {len(ohlcv_list)} OHLCV bars...")
            time.sleep(0.02)
        except Exception as e:
            flush_print(f"  Error fetching OHLCV: {e}")
            break
            
    df = pd.DataFrame(ohlcv_list[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    flush_print(f"Fetching funding rate history for {symbol}...")
    funding_list = []
    current_since_funding = int(df.index[0].timestamp() * 1000)
    end_ms = int(df.index[-1].timestamp() * 1000)
    
    while current_since_funding < end_ms:
        try:
            batch = exchange.fetch_funding_rate_history(symbol, since=current_since_funding, limit=1000)
            if not batch: break
            funding_list.extend(batch)
            current_since_funding = batch[-1]['timestamp'] + 1
            if len(funding_list) % 1000 == 0: flush_print(f"  Fetched {len(funding_list)} funding rows...")
            time.sleep(0.02)
        except Exception as e:
            flush_print(f"  Error fetching funding: {e}")
            break
            
    f_df = pd.DataFrame(funding_list)
    if not f_df.empty:
        f_df['timestamp'] = pd.to_datetime(f_df['timestamp'], unit='ms', utc=True)
        f_df = f_df.set_index('timestamp').sort_index()
        df['funding_rate'] = f_df['fundingRate'].reindex(df.index, method='ffill').fillna(0.0)
    else:
        df['funding_rate'] = 0.0
        
    return df

from concurrent.futures import ThreadPoolExecutor

def download_one_day(dt, symbol, base_url):
    date_str = dt.strftime('%Y-%m-%d')
    url = f"{base_url}/BTCUSDT-bookDepth-{date_str}.zip"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                df = pd.read_csv(z.open(csv_name))
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                return df
    except Exception:
        pass
    return None

def download_binance_vision_depth(date_range):
    flush_print(f"Downloading daily bookDepth summaries for {len(date_range)} days in parallel...")
    base_url = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT"
    all_depth = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda dt: download_one_day(dt, 'BTCUSDT', base_url), date_range))
    
    all_depth = [r for r in results if r is not None]
    if not all_depth: return pd.DataFrame()
    return pd.concat(all_depth).sort_values('timestamp')

def build_bids_asks(ohlcv, depth_df):
    flush_print("Building bids/asks from real depth summaries...")
    depth_df = depth_df.sort_values('timestamp')
    # Unify precision to avoid merge_asof error
    depth_df['timestamp'] = depth_df['timestamp'].dt.as_unit('ns')
    bar_timestamps = ohlcv.index.as_unit('ns')
    unique_depth_ts = depth_df['timestamp'].unique()
    
    matched = pd.merge_asof(
        pd.DataFrame(index=bar_timestamps),
        pd.DataFrame({'match_ts': unique_depth_ts}, index=unique_depth_ts),
        left_index=True, right_index=True, direction='backward'
    )
    
    grouped = depth_df.groupby('timestamp')
    bids_list = []
    asks_list = []
    
    for i, ts in enumerate(bar_timestamps):
        match_ts = matched.iloc[i]['match_ts']
        if pd.isna(match_ts):
            bids_list.append([])
            asks_list.append([])
            continue
            
        snap = grouped.get_group(match_ts)
        close_price = ohlcv.loc[ts, 'close']
        
        b = []
        a = []
        for _, row in snap.iterrows():
            pct = row['percentage']
            price = close_price * (1 + pct / 100.0)
            qty = row['depth']
            if pct < 0: b.append([price, qty])
            else: a.append([price, qty])
        
        b.sort(key=lambda x: x[0], reverse=True)
        a.sort(key=lambda x: x[0])
        bids_list.append(b)
        asks_list.append(a)
        
    return bids_list, asks_list

def main():
    symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
    out_dir = 'backtester_v2/data/raw'
    os.makedirs(out_dir, exist_ok=True)

    for symbol in symbols:
        out_path = f"{out_dir}/{symbol}_real.parquet"
        if os.path.exists(out_path):
            flush_print(f"--- Skipping {symbol}, {out_path} already exists ---")
            continue
            
        flush_print(f"\n--- Processing {symbol} ---")
        # 1. Get OHLCV and Funding from Kraken
        # Kraken symbol mapping
        kraken_symbols = {
            "BTC_USDT": "PI_XBTUSD",
            "ETH_USDT": "PI_ETHUSD",
            "SOL_USDT": "SOL/USD:USD"
        }
        k_sym = kraken_symbols.get(symbol, symbol)
        df = fetch_kraken_futures_data(symbol=k_sym, limit=50000)
        
        # 2. Get Real Depth from Binance Vision
        date_range = pd.date_range(df.index[0].date(), df.index[-1].date())
        depth_df = download_binance_vision_depth(date_range)
        
        if not depth_df.empty:
            bids, asks = build_bids_asks(df, depth_df)
            df['bids'] = bids
            df['asks'] = asks
            
            final_df = df[df['bids'].apply(lambda x: len(x) > 0)].copy()
            out_path = f"{out_dir}/{symbol}_real.parquet"
            final_df.to_parquet(out_path)
            flush_print(f"Saved REAL data to {out_path} ({len(final_df)} bars)")
        else:
            flush_print(f"Failed to download real depth data for {symbol}.")

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import numpy as np
import os
import json
import structlog
from datetime import datetime
from btc_backtester.data.loader import DataLoader
from btc_backtester.models.xgb_strategy import XGBStrategy, walk_forward_cv
from btc_backtester.backtester.engine import BacktestEngine
from btc_backtester.backtester.metrics import calculate_metrics

logger = structlog.get_logger()

def generate_dummy_data(path, freq="15min", start="2022-01-01", end="2024-12-31"):
    print(f"Generating dummy data for {freq} at {path}...")
    dates = pd.date_range(start, end, freq=freq)
    df = pd.DataFrame({
        "open": 50000 + np.cumsum(np.random.randn(len(dates)) * 50),
        "high": 50000 + np.cumsum(np.random.randn(len(dates)) * 50) + 100,
        "low": 50000 + np.cumsum(np.random.randn(len(dates)) * 50) - 100,
        "close": 50000 + np.cumsum(np.random.randn(len(dates)) * 50),
        "volume": np.random.rand(len(dates)) * 1000,
        "funding_rate": np.random.randn(len(dates)) * 0.0001,
        "open_interest": 1000000 + np.cumsum(np.random.randn(len(dates)) * 1000),
    }, index=dates)
    
    # Simple L2 proxy
    def make_l2(price):
        bids = [[price - 10, 10], [price - 20, 20]]
        asks = [[price + 10, 10], [price + 20, 20]]
        return bids, asks

    l2_data = [make_l2(p) for p in df['close']]
    df['bids'] = [x[0] for x in l2_data]
    df['asks'] = [x[1] for x in l2_data]
    df['spot_price'] = df['close'] * 0.9999
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return df

def resample_data(df, freq="4h"):
    print(f"Resampling data to {freq}...")
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "funding_rate": "mean",
        "open_interest": "last",
        "bids": "last", # Snapshot at end of period
        "asks": "last",
    }
    if 'spot_price' in df.columns:
        agg_dict['spot_price'] = 'last'
        
    resampled = df.resample(freq).agg(agg_dict).dropna()
    
    if 'spot_price' not in resampled.columns:
        resampled['spot_price'] = resampled['close']
        
    return resampled

def run_backtest(df, label):
    print(f"\n--- Running Backtest for {label} ---")
    # Increase refit_every for faster execution in this evaluation
    train_window = 90
    refit_every = 30 # Increased from 7
    
    # If we have less than 180 days, reduce window to ensure we get some test periods
    total_days = (df.index.max() - df.index.min()).days
    if total_days < 180:
        train_window = max(30, int(total_days * 0.4))
        refit_every = max(7, int(total_days * 0.2))
        print(f"Adjusted windows for short data: train={train_window}d, refit={refit_every}d")

    results_df = walk_forward_cv(df, train_window=train_window, refit_every=refit_every)
    results_df['spread_bps'] = 2.0
    
    # Detect bars per day for metrics
    freq = pd.Series(df.index).diff().median()
    bars_per_day = int(pd.Timedelta(days=1) / freq)
    periods_per_year = bars_per_day * 365
    
    engine = BacktestEngine()
    final_df = engine.run(results_df)
    
    metrics = calculate_metrics(final_df, periods_per_year=periods_per_year)
    return metrics

def main():
    # 1. 15m Backtest (Real-ish data if exists)
    data_15m_path = "btc_backtester/data/raw/btc_15m.parquet" # Path relative to v0_shinka_evolve
    if os.path.exists(data_15m_path):
        loader = DataLoader(data_15m_path)
        df_15m = loader.load_data()
        # Limit to last 1 year for speed
        df_15m = df_15m.iloc[-35040:] # ~1 year of 15m bars
    else:
        df_15m = generate_dummy_data(data_15m_path, freq="15min")
    
    metrics_15m = run_backtest(df_15m, "15m")
    
    # 2. 4h Backtest (Resampled from 15m)
    df_4h = resample_data(df_15m, "4h")
    metrics_4h = run_backtest(df_4h, "4h")
    
    # 3. 1m Backtest (Generated)
    data_1m_path = "btc_backtester/data/raw/btc_1m_dummy.parquet"
    # Generate 60 days of 1m data
    df_1m = generate_dummy_data(data_1m_path, freq="1min", start="2024-01-01", end="2024-03-01")
    metrics_1m = run_backtest(df_1m, "1m")
    
    all_results = {
        "15m": metrics_15m,
        "4h": metrics_4h,
        "1m": metrics_1m
    }
    
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(json.dumps(all_results, indent=2))
    
    with open("multi_timeframe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from btc_backtester.data.loader import DataLoader
from btc_backtester.models.xgb_strategy import XGBStrategy, walk_forward_cv
from btc_backtester.backtester.engine import BacktestEngine
from btc_backtester.backtester.metrics import calculate_metrics

def generate_dummy_data(path):
    print(f"Generating dummy data at {path}...")
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="15min")
    df = pd.DataFrame({
        "open": 50000 + np.cumsum(np.random.randn(len(dates)) * 50),
        "high": 50000 + np.cumsum(np.random.randn(len(dates)) * 50) + 100,
        "low": 50000 + np.cumsum(np.random.randn(len(dates)) * 50) - 100,
        "close": 50000 + np.cumsum(np.random.randn(len(dates)) * 50),
        "volume": np.random.rand(len(dates)) * 1000,
        "funding_rate": np.random.randn(len(dates)) * 0.0001,
        "open_interest": 1000000 + np.cumsum(np.random.randn(len(dates)) * 1000),
    }, index=dates)
    df['bids'] = [np.array([[df.iloc[i]['close'] - 10, 10], [df.iloc[i]['close'] - 20, 20]]) for i in range(len(df))]
    df['asks'] = [np.array([[df.iloc[i]['close'] + 10, 10], [df.iloc[i]['close'] + 20, 20]]) for i in range(len(df))]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="btc_backtester/data/raw/btc_15m.parquet")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", default="ledger.json")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        generate_dummy_data(args.data)

    loader = DataLoader(args.data)
    df = loader.load_data(args.start, args.end)
    
    print(f"Running Initial Backtest (v0) on {len(df)} bars...")
    
    # Run Walk-Forward CV
    results_df = walk_forward_cv(df)
    
    # We need features like spread_bps for the engine
    # In a real scenario, these would be in the raw data or computed
    # For dummy, let's add them
    results_df['spread_bps'] = 2.0 
    
    engine = BacktestEngine()
    final_df = engine.run(results_df)
    
    metrics = calculate_metrics(final_df)
    metrics["strategy_version"] = "v0"
    metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    print("\nInitial Results (v0):")
    print(json.dumps(metrics, indent=2))
    
    with open(args.out, "w") as f:
        json.dump({"champion": metrics, "history": []}, f, indent=2)

if __name__ == "__main__":
    main()

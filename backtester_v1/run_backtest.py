import argparse
import pandas as pd
import joblib
import os
import sys

# Ensure scripts are importable
sys.path.append(os.path.join(os.getcwd(), 'v0_shinka_evolve'))

from backtester_v1.scripts.feature_engineering import build_features
from backtester_v1.scripts.backtester import load_model, run_backtest
from backtester_v1.scripts.report import compute_metrics, save_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='v0_shinka_evolve/backtester_v1/data/raw/btc_5000_validation.parquet')
    parser.add_argument('--out_dir', default='v0_shinka_evolve/backtester_v1/results')
    args = parser.parse_args()

    # --- Load Raw Data ---
    print(f"Loading raw data from {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df = df.sort_index()

    # --- Load Scaler and Build Features ---
    scaler_path = 'v0_shinka_evolve/backtester_v1/models/scaler_v0.joblib'
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    print("Building features...")
    features = build_features(df, scaler=scaler)
    
    # Align OHLCV to features index
    ohlcv = df.loc[features.index]
    
    # --- Load Model and Run Backtest ---
    model_path = 'v0_shinka_evolve/backtester_v1/models/xgb_regression_v0.json'
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print("Running backtest...")
    state = run_backtest(ohlcv, features, model)
    
    # --- Compute and Save Results ---
    print("Computing metrics...")
    metrics = compute_metrics(state)
    
    print("\n=== BACKTEST RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
        
    print(f"\nSaving results to {args.out_dir}...")
    save_report(metrics, state, args.out_dir)
    print("Done!")

if __name__ == '__main__':
    main()

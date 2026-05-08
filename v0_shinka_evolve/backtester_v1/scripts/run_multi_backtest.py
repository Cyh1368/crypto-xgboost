import os
import pandas as pd
import joblib
import sys
import numpy as np

# Ensure scripts are importable
sys.path.append(os.path.join(os.getcwd(), 'v0_shinka_evolve'))

from backtester_v1.scripts.feature_engineering import build_features
from backtester_v1.scripts.backtester import load_model, run_backtest
from backtester_v1.scripts.report import compute_metrics

def main():
    data_dir = 'v0_shinka_evolve/backtester_v1/data/raw/multi'
    model_path = 'v0_shinka_evolve/backtester_v1/models/xgb_regression_v0.json'
    scaler_path = 'v0_shinka_evolve/backtester_v1/models/scaler_v0.joblib'
    
    # Load Model and Scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    results = []
    
    for file in os.listdir(data_dir):
        if not file.endswith('.parquet'): continue
        
        symbol = file.replace('.parquet', '').replace('_', '/')
        print(f"Backtesting {symbol}...")
        
        path = os.path.join(data_dir, file)
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Build Features
        features = build_features(df, scaler=scaler)
        ohlcv = df.loc[features.index]
        
        # Run Backtest
        state = run_backtest(ohlcv, features, model)
        metrics = compute_metrics(state)
        
        if "Error" in metrics:
            print(f"  No trades for {symbol}")
            continue
            
        metrics['Symbol'] = symbol
        results.append(metrics)
    
    if not results:
        print("No results to report.")
        return
        
    results_df = pd.DataFrame(results)
    
    # Generate Markdown Report
    report_path = 'v0_shinka_evolve/backtester_v1/results/multi_ticker_report.md'
    with open(report_path, 'w') as f:
        f.write("# Multi-Ticker Backtest Report (Last 2 Weeks)\n\n")
        f.write("Summary of performance across 10 major tickers using the universal XGBoost Alpha model.\n\n")
        
        f.write("## 1. Performance Summary Table\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Key Observations\n\n")
        avg_sharpe = results_df['Sharpe Ratio (annual)'].mean()
        avg_win_rate = results_df['Win Rate (%)'].mean()
        total_pnl = results_df['Total Net PnL ($)'].sum()
        
        f.write(f"- **Average Sharpe Ratio**: {avg_sharpe:.3f}\n")
        f.write(f"- **Average Win Rate**: {avg_win_rate:.2f}%\n")
        f.write(f"- **Total Aggregated PnL**: ${total_pnl:.2f}\n")
        f.write(f"- **Best Performer**: {results_df.loc[results_df['Total Net PnL ($)'].idxmax(), 'Symbol']}\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()

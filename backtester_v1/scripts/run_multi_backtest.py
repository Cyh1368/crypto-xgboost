import os
import pandas as pd
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure scripts are importable
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'v0_shinka_evolve'))

from backtester_v1.scripts.feature_engineering import build_features
from backtester_v1.scripts.backtester import load_model, run_backtest
from backtester_v1.scripts.report import compute_metrics

def plot_ticker_results(state, ohlcv_df, symbol, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Subplot 1: PnL
    initial_equity = state.equity_curve[0]
    pnl = np.array(state.equity_curve) - initial_equity
    ax1.plot(state.timestamps, pnl, label='Cumulative Net PnL ($)', color='blue', linewidth=1.5)
    ax1.set_ylabel('PnL ($)')
    ax1.set_title(f'{symbol} Strategy Cumulative PnL Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Subplot 2: Spot Price
    ax2.plot(ohlcv_df.index, ohlcv_df['close'], label='Spot Price', color='gray', alpha=0.6, linewidth=1)
    
    # Plot trades
    for trade in state.trades:
        color = 'green' if trade.direction == 1 else 'red'
        ax2.plot([trade.entry_ts, trade.exit_ts], [trade.entry_price, trade.exit_price], 
                 color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax2.scatter(trade.entry_ts, trade.entry_price, color=color, marker='^' if trade.direction == 1 else 'v', s=50)
        ax2.scatter(trade.exit_ts, trade.exit_price, color='black', marker='o', s=20, alpha=0.5)

    custom_lines = [Line2D([0], [0], color='gray', lw=1),
                    Line2D([0], [0], color='green', lw=2, linestyle='--'),
                    Line2D([0], [0], color='red', lw=2, linestyle='--')]
    ax2.legend(custom_lines, ['Spot Price', 'Long Trade', 'Short Trade'], loc='upper left')
    ax2.set_ylabel('Price (USD)')
    ax2.set_title(f'{symbol} Trade Executions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_symbol = symbol.replace('/', '_')
    plt.savefig(os.path.join(out_dir, f'backtest_visual_{safe_symbol}.png'))
    plt.close()

def main():
    data_dir = 'v0_shinka_evolve/backtester_v1/data/raw/multi'
    model_path = 'v0_shinka_evolve/backtester_v1/models/xgb_regression_v0.json'
    scaler_path = 'v0_shinka_evolve/backtester_v1/models/scaler_v0.joblib'
    results_dir = 'v0_shinka_evolve/backtester_v1/results'
    
    os.makedirs(results_dir, exist_ok=True)
    
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
            df.index = pd.to_datetime(df.index).tz_localize('UTC')
        else:
            df.index = pd.to_datetime(df.index).tz_convert('UTC')
        
        # Build Features
        features = build_features(df, scaler=scaler)
        ohlcv = df.loc[features.index]
        
        # Ensure indices are tz-naive for backtester logic and plotting
        features.index = features.index.tz_localize(None)
        ohlcv.index = ohlcv.index.tz_localize(None)
        
        # Run Backtest
        state = run_backtest(ohlcv, features, model)

        metrics = compute_metrics(state)
        
        if "Error" in metrics:
            print(f"  No trades for {symbol}")
            continue
            
        metrics['Symbol'] = symbol
        results.append(metrics)
        
        # Generate Plot
        plot_ticker_results(state, ohlcv, symbol, results_dir)
        print(f"  Plot saved for {symbol}")
    
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

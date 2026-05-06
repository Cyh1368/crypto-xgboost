import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from btc_backtester.data.loader import DataLoader
from btc_backtester.models.xgb_strategy import XGBStrategy, walk_forward_cv
from btc_backtester.backtester.engine import BacktestEngine
from btc_backtester.backtester.metrics import calculate_metrics

def run_backtest_for_symbol(file_path):
    loader = DataLoader(file_path)
    df = loader.load_data()
    
    print(f"Running backtest for {os.path.basename(file_path)}...")
    
    # Run Walk-Forward CV (simplified for fast result on 500 bars)
    # 500 bars is about 5 days. train_window=90 days is too big.
    # Let's reduce window for this small sample.
    results_df = walk_forward_cv(df, train_window=2, refit_every=1)
    
    if results_df.empty:
        print(f"No results for {file_path}")
        return None
    
    # Ensure spread_bps and other required fields
    results_df['spread_bps'] = 1.0 
    
    engine = BacktestEngine()
    final_df = engine.run(results_df)
    
    return final_df

def plot_results(df, symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Prices
    ax1.plot(df.index, df['close'], label='Futures Price', color='blue', alpha=0.7)
    if 'spot_price' in df.columns:
        ax1.plot(df.index, df['spot_price'], label='Spot Price', color='orange', alpha=0.5)
    
    # Mark actions
    longs = df[df['signal'] == 1]
    shorts = df[df['signal'] == -1]
    
    ax1.scatter(longs.index, longs['close'], marker='^', color='green', label='Long', s=50)
    ax1.scatter(shorts.index, shorts['close'], marker='v', color='red', label='Short', s=50)
    
    ax1.set_title(f"Backtest: {symbol} - Price and Actions")
    ax1.legend()
    ax1.grid(True)
    
    # Plot PnL / Equity
    ax2.plot(df.index, df['equity'], label='Equity', color='purple')
    ax2.set_title(f"Backtest: {symbol} - Equity Curve")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = f"btc_backtester/data/processed/plot_{symbol.replace('/', '_')}.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

def main():
    data_dir = 'btc_backtester/data/raw/real'
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    for f in files:
        symbol = f.replace('.parquet', '').replace('_', '/')
        final_df = run_backtest_for_symbol(os.path.join(data_dir, f))
        if final_df is not None:
            plot_results(final_df, symbol)

if __name__ == "__main__":
    main()

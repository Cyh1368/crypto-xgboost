import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_backtest_results(file_path, output_path):
    df = pd.read_parquet(file_path)
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Spot Price and Signals
    ax1.plot(df.index, df['close'], label='BTC Close Price', color='blue', alpha=0.6)
    
    # To avoid clutter, only plot markers where signal changes
    # Long entries (signal changes to 1)
    longs = df[df['signal'].diff() > 0]
    ax1.scatter(longs.index, longs['close'], marker='^', color='green', label='Long Entry', s=50, zorder=5)
    
    # Short entries (signal changes to -1)
    shorts = df[df['signal'].diff() < 0]
    ax1.scatter(shorts.index, shorts['close'], marker='v', color='red', label='Short Entry', s=50, zorder=5)
    
    ax1.set_title('BTC/USD Price and Trade Signals')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity Curve (PnL)
    ax2.plot(df.index, df['equity'], label='Portfolio Equity', color='black')
    ax2.set_title('Equity Curve (Cumulative PnL)')
    ax2.set_ylabel('Equity (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position Size
    ax3.fill_between(df.index, df['pos_size'], 0, color='gray', alpha=0.5, label='Position Size')
    ax3.set_title('Position Sizing')
    ax3.set_ylabel('Size (Fraction of NAV)')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    results_file = 'btc_backtester/results/paper_framework_test_results.parquet'
    output_file = 'btc_backtester/results/backtest_visualization.png'
    
    if os.path.exists(results_file):
        plot_backtest_results(results_file, output_file)
    else:
        print(f"Results file {results_file} not found.")

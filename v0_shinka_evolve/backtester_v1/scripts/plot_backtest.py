import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main():
    results_dir = 'v0_shinka_evolve/backtester_v1/results'
    trades_path = f"{results_dir}/trades.csv"
    equity_path = f"{results_dir}/equity_curve.csv"
    data_path = 'v0_shinka_evolve/backtester_v1/data/raw/btc_5000_validation.parquet'

    if not all(os.path.exists(p) for p in [trades_path, equity_path, data_path]):
        print("Error: Results or raw data not found. Run the backtest first.")
        return

    # Load Data
    trades_df = pd.read_csv(trades_path, parse_dates=['entry_ts', 'exit_ts'])
    equity_df = pd.read_csv(equity_path, parse_dates=['timestamp'])
    spot_df = pd.read_parquet(data_path)
    
    # Standardize timezones
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp']).dt.tz_localize(None)
    trades_df['entry_ts'] = pd.to_datetime(trades_df['entry_ts']).dt.tz_localize(None)
    trades_df['exit_ts'] = pd.to_datetime(trades_df['exit_ts']).dt.tz_localize(None)
    spot_df.index = pd.to_datetime(spot_df.index).tz_localize(None)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Subplot 1: Equity Curve / PnL
    initial_equity = equity_df['equity'].iloc[0]
    pnl = equity_df['equity'] - initial_equity
    ax1.plot(equity_df['timestamp'], pnl, label='Cumulative Net PnL ($)', color='blue', linewidth=1.5)
    ax1.set_ylabel('PnL ($)')
    ax1.set_title('Strategy Cumulative PnL Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Subplot 2: Spot Price and Trade Markers
    ax2.plot(spot_df.index, spot_df['close'], label='BTC Spot Price', color='gray', alpha=0.6, linewidth=1)
    
    # Plot trade executions
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['direction'] == 1 else 'red'
        label = 'Long' if trade['direction'] == 1 else 'Short'
        
        # Draw dashed line between entry and exit
        ax2.plot([trade['entry_ts'], trade['exit_ts']], 
                 [trade['entry_price'], trade['exit_price']], 
                 color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Markers for entry and exit
        ax2.scatter(trade['entry_ts'], trade['entry_price'], color=color, marker='^' if trade['direction'] == 1 else 'v', s=50)
        ax2.scatter(trade['exit_ts'], trade['exit_price'], color='black', marker='o', s=20, alpha=0.5)

    # Custom legend for trades to avoid duplicates
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=1),
                    Line2D([0], [0], color='green', lw=2, linestyle='--'),
                    Line2D([0], [0], color='red', lw=2, linestyle='--')]
    ax2.legend(custom_lines, ['Spot Price', 'Long Trade', 'Short Trade'], loc='upper left')
    
    ax2.set_ylabel('BTC Price (USD)')
    ax2.set_xlabel('Time')
    ax2.set_title('Trade Executions on Spot Price')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{results_dir}/backtest_visual.png"
    plt.savefig(plot_path)
    print(f"Backtest visualization saved to {plot_path}")

if __name__ == "__main__":
    main()

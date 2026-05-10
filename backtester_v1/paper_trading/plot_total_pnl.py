import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = 'backtester_v1/paper_trading/results'
INPUT_FILE = os.path.join(BASE_DIR, 'total_pnl.csv')
OUTPUT_PLOT = os.path.join(BASE_DIR, 'total_pnl_plot.png')

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Has paper trading started?")
        return

    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        print("Error: total_pnl.csv is empty.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['total_pnl'], marker='o', linestyle='-', color='blue', label='Total Net PnL ($)')
    
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('Real-Time Paper Trading: Total Cumulative PnL Over Time', fontsize=14)
    plt.xlabel('Time (UTC)')
    plt.ylabel('PnL ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis for readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT)
    print(f"Total PnL plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()

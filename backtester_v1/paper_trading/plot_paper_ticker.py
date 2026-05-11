import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = 'backtester_v1/paper_trading/results'
INITIAL_COIN_EQUITY = 10.0

def plot_paper_ticker_advanced(symbol):
    safe_name = symbol.replace('/', '_')
    trades_path = os.path.join(BASE_DIR, f'trades_{safe_name}.csv')
    total_pnl_path = os.path.join(BASE_DIR, 'total_pnl.csv')

    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return

    df = pd.read_csv(trades_path)
    if len(df) < 2:
        print(f"Error: Not enough data in {trades_path} to plot.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Reconstruct Equity Curve (Realized + Unrealized)
    # We need to estimate qty and entry_price to find unrealized
    # We'll use the total_pnl.csv to get the total equity context for sizing
    total_equity_map = {}
    if os.path.exists(total_pnl_path):
        total_df = pd.read_csv(total_pnl_path)
        total_df['timestamp'] = pd.to_datetime(total_df['timestamp'])
        total_equity_map = total_df.set_index('timestamp')['total_equity'].to_dict()

    realized_pnl = df['pnl'].cumsum()
    equity_curve = []
    positions = df['position'].values
    prices = df['price'].values
    actions = df['action'].values
    
    current_entry_price = 0.0
    current_qty = 0.0
    current_notional = 0.0
    
    sizes = [] # For Subplot 3
    
    for i in range(len(df)):
        price_now = prices[i]
        pos = positions[i]
        action = actions[i]
        ts = df.loc[i, 'timestamp']
        
        # Detect Entry
        if "ENTER" in action:
            current_entry_price = price_now
            # Estimate notional: backtester used out['position_size'] * Total_Equity / 10
            # We don't have out['position_size'] saved, but we can assume ~0.15
            # Or we can just use the PnL to back-calculate if we had a trade.
            # For now, let's use a default size or look at the PnL of the first exit.
            # Better: The PaperTrader logs PnL at EXIT. 
            # net_pnl = qty * (exit - entry) * dir - fees
            # net_pnl + fees = qty * (exit - entry) * dir
            # qty = (net_pnl + fees) / ((exit - entry) * dir)
            
            # Since we can't easily get it here without looking ahead, 
            # let's assume a standard 15% sizing of the $100 allocation as a fallback.
            current_notional = total_equity_map.get(ts, 1000.0) * 0.15 / 10.0
            current_qty = current_notional / price_now
        
        unrealized = 0.0
        if pos != 0 and current_qty > 0:
            unrealized = current_qty * (price_now - current_entry_price) * pos
            # Total Portfolio = 10 * Symbol Equity
            # Symbol Equity = INITIAL_COIN_EQUITY + realized_pnl + unrealized
            sym_equity = INITIAL_COIN_EQUITY + realized_pnl[i] + unrealized
            notional = current_qty * price_now
            size_pct = (notional / (sym_equity * 10.0)) * 100.0
            sizes.append(size_pct * pos)
        else:
            sizes.append(0.0)
            if pos == 0:
                current_qty = 0.0
                
        equity_curve.append(INITIAL_COIN_EQUITY + realized_pnl[i] + unrealized)

    # 1. PnL Plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24), sharex=False)
    fig.suptitle(f"Paper Trading: {symbol}, Initial Portfolio = ${INITIAL_COIN_EQUITY}", fontsize=20, fontweight='bold')
    
    ax1.plot(df['timestamp'], np.array(equity_curve) - INITIAL_COIN_EQUITY, label='Cumulative Net PnL ($)', color='blue', linewidth=1.5)
    ax1.set_ylabel('PnL ($)')
    ax1.set_title('Cumulative PnL Over Time (Realized + Unrealized)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 2. Price and Trades
    ax2.plot(df['timestamp'], df['price'], label='Spot Price', color='gray', alpha=0.6, linewidth=1)
    for i in range(len(df)):
        if "ENTER" in actions[i]:
            color = 'green' if positions[i] == 1 else 'red'
            ax2.scatter(df.loc[i, 'timestamp'], df.loc[i, 'price'], color=color, marker='^' if positions[i] == 1 else 'v', s=100, zorder=5)
        elif "EXIT" in actions[i]:
            ax2.scatter(df.loc[i, 'timestamp'], df.loc[i, 'price'], color='black', marker='o', s=30, alpha=0.5, zorder=5)

    ax2.set_ylabel('Price (USD)')
    ax2.set_title('Spot Price & Trade Markers')
    ax2.grid(True, alpha=0.3)

    # 3. Position Sizing
    ax3.bar(df['timestamp'], sizes, width=0.005, color=['green' if s > 0 else 'red' for s in sizes], alpha=0.7)
    ax3.set_ylabel('Size (% of Total Portfolio)')
    ax3.set_title('Position Sizing Over Time')
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.grid(True, alpha=0.3)

    # 4. Scatter Plot (Prediction Accuracy)
    # actual_ratio = price[t+1] / price[t]
    actual_ratios = df['price'].shift(-1) / df['price']
    pred_ratios = 1.0 + df['prediction']
    
    clean_df = pd.DataFrame({'actual': actual_ratios, 'predicted': pred_ratios, 'action': actions}).dropna()
    if not clean_df.empty:
        colors = clean_df['action'].apply(lambda x: 'green' if 'ENTER_LONG' in x else ('red' if 'ENTER_SHORT' in x else ('orange' if 'EXIT' in x else 'black')))
        ax4.scatter(clean_df['actual'], clean_df['predicted'], c=colors, s=5, alpha=0.6)
        
        # y=x line
        min_val = min(clean_df['actual'].min(), clean_df['predicted'].min())
        max_val = max(clean_df['actual'].max(), clean_df['predicted'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        
        ax4.set_xlabel('Actual Price Ratio')
        ax4.set_ylabel('Predicted Price Ratio')
        ax4.set_title('Predictive Accuracy (Next-Bar)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(BASE_DIR, f'paper_advanced_{safe_name}.png')
    plt.savefig(output_path)
    print(f"Advanced paper trading plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', help='Ticker symbol, e.g., BTC/USDT')
    args = parser.parse_args()
    plot_paper_ticker_advanced(args.ticker)

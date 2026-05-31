import os
import pandas as pd
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import xgboost as xgb

# Add project root to path
sys.path.append(os.getcwd())

from backtester_v2.scripts.feature_engineering import build_features
from backtester_v1.scripts.backtester import load_model, run_backtest, predict_ratios
from backtester_v1.scripts.report import compute_metrics

def plot_advanced_ticker_results(state, ohlcv_df, features_df, model, calibration_factor, symbol, out_dir):
    # Calculate Sharpe for title
    metrics = compute_metrics(state)
    sharpe = metrics.get('Sharpe Ratio (annual)', 0)
    
    # Calculate Predicted vs Actual Ratios for Scatter Plot
    # predict_ratios returns 1.0 + (bps * calibration * BPS_TO_RATIO)
    pred_ratios = predict_ratios(model, features_df, calibration_factor)
    
    # Target: next bar ratio
    actual_ratios = ohlcv_df['close'].shift(-1) / ohlcv_df['close']
    
    # Create DataFrame for scatter plotting
    scatter_df = pd.DataFrame({
        'predicted': pred_ratios,
        'actual': actual_ratios,
        'action': 'none',
        'timestamp': ohlcv_df.index
    })
    
    # Mark actions
    for trade in state.trades:
        scatter_df.loc[scatter_df['timestamp'] == trade.entry_ts, 'action'] = 'entry'
        scatter_df.loc[scatter_df['timestamp'] == trade.exit_ts, 'action'] = 'exit'
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24), sharex=False)
    
    # Set overall title
    initial_equity = state.equity_curve[0]
    fig.suptitle(f"{symbol}, initial portfolio = $10, Sharpe = {sharpe:.2f}", fontsize=20, fontweight='bold')
    
    # Subplot 1: PnL
    pnl = np.array(state.equity_curve) - initial_equity
    ax1.plot(state.timestamps, pnl, label='Cumulative Net PnL ($)', color='blue', linewidth=1.5)
    ax1.set_ylabel('PnL ($)')
    ax1.set_title(f'Strategy Cumulative PnL Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Subplot 2: Spot Price and Trade Markers
    ax2.plot(ohlcv_df.index, ohlcv_df['close'], label='Spot Price', color='gray', alpha=0.6, linewidth=1)
    for trade in state.trades:
        color = 'green' if trade.direction == 1 else 'red'
        # Dash line between entry and exit
        ax2.plot([trade.entry_ts, trade.exit_ts], [trade.entry_price, trade.exit_price], 
                 color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Markers for entry and exit
        ax2.scatter(trade.entry_ts, trade.entry_price, color=color, marker='^' if trade.direction == 1 else 'v', s=20, zorder=5)
        ax2.scatter(trade.exit_ts, trade.exit_price, color='black', marker='o', s=20, alpha=0.5, zorder=5)

    custom_lines = [Line2D([0], [0], color='gray', lw=1),
                    Line2D([0], [0], color='green', lw=2, linestyle='--'),
                    Line2D([0], [0], color='red', lw=2, linestyle='--')]
    ax2.legend(custom_lines, ['Spot Price', 'Long Trade', 'Short Trade'], loc='upper left')
    ax2.set_ylabel('Price (USD)')
    ax2.set_title(f'Trade Executions')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Position Sizing (Bar Chart)
    # size = (Notional / Total Portfolio Equity) * 100
    sizes = []
    for ts, pos, eq in zip(state.timestamps, state.positions, state.equity_curve):
        if pos == 0:
            sizes.append(0.0)
        else:
            active_trade = None
            for t in state.trades:
                if t.entry_ts <= ts <= t.exit_ts:
                    active_trade = t
                    break
            if active_trade:
                notional = active_trade.qty_btc * active_trade.entry_price
                size_pct = (notional / eq) * 100.0
                sizes.append(size_pct * pos)
            else:
                sizes.append(0.0)

    ax3.bar(state.timestamps, sizes, width=0.01, color=['green' if s > 0 else 'red' for s in sizes], alpha=0.7)
    ax3.set_ylabel('Size (% of Total Portfolio)')
    ax3.set_title(f'Strategy Position Sizing Over Time')
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Advanced Scatter Plot (Predicted vs Actual)
    clean_scatter = scatter_df.dropna()
    colors = clean_scatter['action'].map({'none': 'black', 'entry': 'green', 'exit': 'red'})
    dot_size = 2
    
    x = clean_scatter['actual'].values
    y = clean_scatter['predicted'].values
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = slope * x_range + intercept
        
        y_pred = slope * x + intercept
        n = len(x)
        mse = np.sum((y - y_pred)**2) / (n - 2)
        x_mean = np.mean(x)
        Sxx = np.sum((x - x_mean)**2)
        stdev = np.sqrt(mse * (1.0/n + (x_range - x_mean)**2 / Sxx))
        ci = 1.96 * stdev
        
        ax4.fill_between(x_range, y_range - ci, y_range + ci, color='yellow', alpha=0.3, edgecolor='orange', label='95% CI', zorder=1)
        ax4.plot(x_range, y_range, color='blue', label='Regression Line', linewidth=2, zorder=2)
    
    ax4.scatter(clean_scatter['actual'], clean_scatter['predicted'], c=colors, s=dot_size, alpha=0.5, zorder=3)
    
    min_val = min(clean_scatter['actual'].min(), clean_scatter['predicted'].min())
    max_val = max(clean_scatter['actual'].max(), clean_scatter['predicted'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x (Parity)', zorder=4)
    
    ax4.set_xlabel('Actual Price Ratio (Price_t+1 / Price_t)')
    ax4.set_ylabel('Predicted Price Ratio')
    ax4.set_title(f'Predictive Accuracy & Trade Distribution')
    ax4.grid(True, alpha=0.3)
    
    scatter_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='No Action'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Entry'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Exit'),
                      Line2D([0], [0], color='blue', lw=2, label='Regression Line'),
                      Patch(facecolor='lightyellow', edgecolor='none', alpha=0.8, label='95% CI')]
    ax4.legend(handles=scatter_legend, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, f'advanced_backtest_regression_as_classifier.png'))
    plt.close()

def main():
    data_path = 'backtester_v2/data/raw/BTC_USDT_50k.parquet'
    model_path = 'backtester_v1/models/xgb_regression_v1.json'
    scaler_path = 'backtester_v1/models/scaler_v1.joblib'
    calib_path = 'backtester_v1/models/calibration_v1.joblib'
    results_dir = 'backtester_v2/results'
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    print("Loading model and scaler...")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    calibration_factor = joblib.load(calib_path)
    
    print("Building features...")
    features = build_features(df, scaler=scaler)
    ohlcv = df.loc[features.index]
    
    if features.index.tz:
        features.index = features.index.tz_localize(None)
    if ohlcv.index.tz:
        ohlcv.index = ohlcv.index.tz_localize(None)
            
    print("Running backtest...")
    # Initial equity $10 as requested in prompt ($10 per coin vibe)
    state = run_backtest(ohlcv, features, model, initial_equity=10.0, calibration_factor=calibration_factor)

    print("Generating advanced plot...")
    plot_advanced_ticker_results(state, ohlcv, features, model, calibration_factor, "BTC_USDT", results_dir)
    print(f"Advanced Plot saved to {results_dir}/advanced_backtest_regression_as_classifier.png")

if __name__ == "__main__":
    main()

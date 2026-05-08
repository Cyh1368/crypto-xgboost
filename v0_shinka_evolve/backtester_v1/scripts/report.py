import pandas as pd
import numpy as np

def compute_metrics(state, risk_free_rate_annual=0.05) -> dict:
    if not state.trades:
        return {"Error": "No trades executed"}
        
    trades_df = pd.DataFrame([vars(t) for t in state.trades])
    equity = np.array(state.equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    total_pnl = trades_df['net_pnl_usd'].sum()
    total_trades = len(trades_df)
    win_trades = (trades_df['net_pnl_usd'] > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    avg_win = trades_df.loc[trades_df['net_pnl_usd'] > 0, 'net_pnl_usd'].mean() if win_trades > 0 else 0
    avg_loss = trades_df.loc[trades_df['net_pnl_usd'] < 0, 'net_pnl_usd'].mean() if (total_trades - win_trades) > 0 else 0
    profit_factor = abs(avg_win * win_trades) / abs(avg_loss * (total_trades - win_trades) + 1e-9)
    
    # Risk Metrics
    bars_per_year = 365 * 24 * 4 # 15-min bars
    rf_per_bar = (1 + risk_free_rate_annual) ** (1 / bars_per_year) - 1
    excess_returns = returns - rf_per_bar
    
    sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(bars_per_year)
    
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    initial_equity = state.equity_curve[0] if state.equity_curve else 100000.0
    calmar = (total_pnl / initial_equity) / abs(max_drawdown + 1e-9)
    
    downside = excess_returns[excess_returns < 0]
    sortino = np.mean(excess_returns) / (np.std(downside) + 1e-9) * np.sqrt(bars_per_year)
    
    total_fees = trades_df['fees_usd'].sum()
    
    return {
        'Total Net PnL ($)': round(total_pnl, 2),
        'Total Trades': total_trades,
        'Win Rate (%)': round(win_rate * 100, 2),
        'Profit Factor': round(profit_factor, 3),
        'Sharpe Ratio (annual)': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Max Drawdown (%)': round(max_drawdown * 100, 2),
        'Calmar Ratio': round(calmar, 3),
        'Avg Win ($)': round(avg_win, 2),
        'Avg Loss ($)': round(avg_loss, 2),
        'Total Fees Paid ($)': round(total_fees, 2),
    }

def save_report(metrics: dict, state, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Metrics
    with open(f"{results_dir}/backtest_report.md", "w") as f:
        f.write("# Backtest Report\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v} |\n")
            
    # Save Trades
    if state.trades:
        trades_df = pd.DataFrame([vars(t) for t in state.trades])
        trades_df.to_csv(f"{results_dir}/trades.csv", index=False)
        
    # Save Equity Curve
    equity_df = pd.DataFrame({
        'timestamp': state.timestamps,
        'equity': state.equity_curve
    })
    equity_df.to_csv(f"{results_dir}/equity_curve.csv", index=False)
    
import os

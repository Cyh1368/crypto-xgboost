import numpy as np
import pandas as pd

def calculate_sharpe(returns: pd.Series, risk_free_rate=0.0, periods_per_year=35040) -> float:
    # 35040 = 365 * 24 * 4 (15-min bars)
    excess_returns = returns - risk_free_rate / periods_per_year
    if len(excess_returns) < 2 or excess_returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std())

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())

def calculate_calmar(sharpe: float, max_dd: float, periods_per_year=35040) -> float:
    # Simplified Calmar: Annualized Return / Max DD
    # We'll use Sharpe as a proxy if we don't have annualized return directly
    # Better: calculate annualized return first
    return abs(sharpe / max_dd) if max_dd != 0 else 0.0

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    df must have 'pnl' and 'equity' columns.
    """
    returns = df['pnl']
    equity = df['equity']
    
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(equity)
    
    win_rate = (returns > 0).mean()
    avg_trade = returns[returns != 0].mean()
    trades_total = (df['signal'].diff() != 0).sum()
    
    return {
        "sharpe_annual": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": calculate_calmar(sharpe, max_dd),
        "win_rate": win_rate,
        "avg_trade_pct": avg_trade * 100,
        "trades_total": int(trades_total),
    }

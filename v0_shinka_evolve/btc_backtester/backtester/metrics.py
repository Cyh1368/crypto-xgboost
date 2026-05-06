import numpy as np
import pandas as pd

def calculate_sharpe(returns: pd.Series, risk_free_rate=0.0, periods_per_year=35040) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    if len(excess_returns) < 2 or excess_returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std())

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / (running_max + 1e-9)
    return float(drawdown.min())

def calculate_calmar(sharpe: float, max_dd: float) -> float:
    return abs(sharpe / max_dd) if max_dd != 0 else 0.0

def calculate_metrics(df: pd.DataFrame, periods_per_year=35040) -> dict:
    """
    df must have 'pnl' and 'equity' columns.
    """
    returns = df['pnl']
    equity = df['equity']
    
    sharpe = calculate_sharpe(returns, periods_per_year=periods_per_year)
    max_dd = calculate_max_drawdown(equity)
    
    win_rate = (returns > 0).mean()
    avg_trade = returns[returns != 0].mean()
    trades_total = (df['signal'].diff() != 0).sum()
    
    # Regime awareness: Tag each bar with a volatility regime
    vol = returns.rolling(20).std()
    try:
        regime = pd.qcut(vol, 3, labels=False, duplicates='drop')
        regime_names = {0: 'low', 1: 'mid', 2: 'high'}
        
        regime_sharpe = {}
        for r_val, r_name in regime_names.items():
            mask = regime == r_val
            if mask.any():
                regime_sharpe[f"sharpe_{r_name}_vol"] = calculate_sharpe(returns[mask], periods_per_year=periods_per_year)
    except:
        regime_sharpe = {}

    metrics = {
        "sharpe_annual": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": calculate_calmar(sharpe, max_dd),
        "win_rate": float(win_rate),
        "avg_trade_pct": float(avg_trade * 100) if not np.isnan(avg_trade) else 0.0,
        "trades_total": int(trades_total),
    }
    metrics.update(regime_sharpe)
    return metrics

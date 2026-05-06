import pandas as pd
import numpy as np
from .metrics import calculate_metrics
from .portfolio import PortfolioManager

class BacktestEngine:
    def __init__(self, initial_nav=100000.0):
        self.initial_nav = initial_nav
        self.portfolio = PortfolioManager(initial_nav)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects df with:
        - close, returns
        - signal, prob
        - spread_bps
        - funding_rate
        - atr (optional)
        """
        # 1. Position sizing (vectorized)
        # Note: In a real event-driven engine, this happens bar by bar
        df['pos_size'] = df.apply(
            lambda row: self.portfolio.size_position(row['prob'], row['signal'], 0, row['close']),
            axis=1
        )
        
        # 2. PnL from price move
        # Shift pos_size because we enter at close of bar t and hold for t+1 return
        df['returns'] = df['close'].pct_change().shift(-1)
        df['pnl_price'] = df['pos_size'] * df['returns']
        
        # 3. Slippage & Fees
        # Fee when changing position
        df['pos_diff'] = df['pos_size'].diff().abs()
        # Slippage: 0.5 * spread_bps + 0.5 bps
        df['slippage_bps'] = 0.5 * df['spread_bps'] + 0.5
        df['slippage_cost'] = df['pos_diff'] * (df['slippage_bps'] / 10000)
        
        # 4. Funding Costs
        # Funding rate is usually 8h rate. 15m is 1/32 of 8h.
        df['funding_cost'] = df['pos_size'] * (df['funding_rate'] / 32)
        
        # 5. Total PnL
        df['pnl'] = df['pnl_price'] - df['slippage_cost'] - df['funding_cost']
        df['equity'] = self.initial_nav * (1 + df['pnl'].cumsum())
        
        # 6. Check Risk (simplified)
        self.portfolio.check_risk(df)
        if self.portfolio.trading_halted:
            # Zero out PnL after halt
            halt_idx = (df['equity'].rolling(480).max() - df['equity']) / df['equity'].rolling(480).max() > 0.08
            # This is a bit tricky in vectorized mode, but let's just find the first halt
            first_halt = halt_idx.idxmax() if halt_idx.any() else None
            if first_halt:
                df.loc[first_halt:, 'pnl'] = 0
                df['equity'] = self.initial_nav * (1 + df['pnl'].cumsum())
        
        return df

import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(self, initial_nav=100000.0, max_single_pos=0.20):
        self.initial_nav = initial_nav
        self.nav = initial_nav
        self.max_single_pos = max_single_pos
        self.stop_loss_mult = 1.5
        self.take_profit_mult = 2.5
        self.trading_halted = False

    def size_position(self, prob: float, signal: int, atr: float, price: float) -> float:
        """
        Returns position size as fraction of NAV.
        """
        if self.trading_halted:
            return 0.0
            
        SIZE_SCALE = 0.10
        # Position scaling by |prob - 0.5|
        size = signal * SIZE_SCALE * 2 * (abs(prob - 0.5))
        
        # Clip to max single position
        size = np.clip(size, -self.max_single_pos, self.max_single_pos)
        return size

    def check_risk(self, df: pd.DataFrame):
        """
        Max drawdown kill-switch: halt trading if rolling 5-day DD > 8%
        """
        # 5 days = 5 * 96 = 480 bars
        if len(df) < 480:
            return
            
        rolling_max = df['equity'].rolling(480).max()
        rolling_dd = (df['equity'] - rolling_max) / rolling_max
        if (rolling_dd < -0.08).any():
            self.trading_halted = True

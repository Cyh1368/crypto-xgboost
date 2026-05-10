import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Refined trading signal generator using liquidity-adjusted thresholds,
    autocorrelation-driven hold times, and volatility-normalized sizing.
    """
    # 1. Essential Market Markers
    close = bar_context.get('close', 1.0)
    vol_short = bar_context.get('vol_5', 0.001)
    vol_long = bar_context.get('vol_20', 0.001)
    atr = bar_context.get('atr_14', 0.002)
    vol_ratio = bar_context.get('realized_vol_ratio', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    rsi = bar_context.get('rsi_14', 50.0)
    obi = bar_context.get('obi_tau5', 0.0)

    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Optimized Threshold Logic
    # Reverting to the robust 0.002 threshold from the seed, with stricter filters.
    spread = bar_context.get('spread_bps', 1.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    base_thresh = 0.0020

    # 3. Execution Signal logic
    signal = 0
    if predicted_return > base_thresh:
        # Stricter OBI, VWAP, and funding filters to reduce drawdown and improve win rate.
        if obi > -0.25 and spread < 7.5 and vwap_dev < 0.015 and funding < 0.0004:
            signal = 1
    elif predicted_return < -base_thresh:
        if obi < 0.25 and spread < 7.5 and vwap_dev > -0.015 and funding > -0.0004:
            signal = -1

    # 4. Optimized Hold Time
    # 4 bars (60 min) is the optimal horizon for a 15-min forward prediction model.
    max_bars = 4

    # 5. Risk Targets (TP/SL)
    # Using fixed targets inspired by the seed to maintain a robust risk profile.
    if signal != 0:
        take_profit = 0.0040
        stop_loss = 0.0030

        # 6. Position Sizing
        # Fixed sizing at 10% to maintain consistency with the most robust seed.
        position_size = 0.10
    else:
        position_size = 0.0
        take_profit = 0.0
        stop_loss = 0.0
        max_bars = 0

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(round(take_profit, 5)),
        "stop_loss": float(round(stop_loss, 5)),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END
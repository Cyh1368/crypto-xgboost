import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Advanced signal generator combining high-threshold entry logic with
    microstructure filters and session-based hold adjustments.
    """
    # 1. Extract Core Context
    obi_10 = bar_context.get('obi_tau10', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    rsi = bar_context.get('rsi_14', 50.0)
    funding_rate = bar_context.get('funding_rate', 0.0)
    vol_20 = bar_context.get('vol_20', 0.01)

    # 2. Dynamic Threshold Logic
    # Lower threshold to increase trade frequency, adjusted slightly by funding costs
    BASE_THRESHOLD = 0.0017
    funding_adj = funding_rate * 0.1
    long_threshold = BASE_THRESHOLD + funding_adj
    short_threshold = -BASE_THRESHOLD + funding_adj

    # 3. Signal Determination with Microstructure & RSI Filters
    signal = 0
    if predicted_return > long_threshold:
        # Relaxed filters to increase trade frequency while avoiding extremes
        if obi_10 > -0.85 and spread < 10.0 and rsi < 80:
            signal = 1
    elif predicted_return < short_threshold:
        if obi_10 < 0.85 and spread < 10.0 and rsi > 20:
            signal = -1

    # 4. Risk Management (Optimized for 15-min forward returns)
    take_profit = 0.0042
    stop_loss = 0.0030

    # 5. Position Sizing
    # Increased to 0.12 to capitalize on high win rate, with a vol-based reduction
    if signal != 0:
        position_size = 0.12
        if vol_20 > 0.02:
            position_size *= 0.8
    else:
        position_size = 0.0

    # 6. Time-Based Exit
    max_bars = 4

    return {
        "signal": int(signal),
        "position_size": float(position_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END
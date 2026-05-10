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
    obi = bar_context.get('obi_tau5', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    trend = bar_context.get('trend_strength', 0.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    is_us_session = bar_context.get('is_us_session', False)
    
    # 2. Dynamic Threshold Logic
    # Anchor to the successful 0.002 threshold from the top-performing seed
    BASE_THRESHOLD = 0.0020
    
    # If the prediction aligns with the trend, we are slightly more aggressive
    if (predicted_return > 0 and trend > 0.4) or (predicted_return < 0 and trend < -0.4):
        effective_threshold = BASE_THRESHOLD * 0.9
    else:
        effective_threshold = BASE_THRESHOLD

    # 3. Signal Determination with Microstructure & Extension Filters
    signal = 0
    if predicted_return > effective_threshold:
        # Long Filter: OBI not heavily against us, spread not too wide, not overextended
        if obi > -0.5 and spread < 9.0 and vwap_dev < 0.02:
            signal = 1
    elif predicted_return < -effective_threshold:
        # Short Filter: OBI not heavily against us, spread not too wide, not overextended
        if obi < 0.5 and spread < 9.0 and vwap_dev > -0.02:
            signal = -1

    # 4. Risk Management (Anchored to the successful 0.004/0.003 seed)
    # Fixed TP/SL has proven robust for this model's 15-min predictions
    take_profit = 0.0040
    stop_loss = 0.0030
    
    # 5. Position Sizing
    # Slightly increased from 0.10 to 0.11 to maximize the high win-rate potential
    position_size = 0.11 if signal != 0 else 0.0
    
    # 6. Dynamic Hold Time (Session Dependent)
    # US sessions often show more follow-through, Asia/Europe can mean-revert
    if is_us_session:
        max_bars = 5
    else:
        max_bars = 4

    return {
        "signal": int(signal),
        "position_size": float(position_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END

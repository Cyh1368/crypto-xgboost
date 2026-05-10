import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Expert trading signal generator utilizing microstructure filters and
    dynamic risk management.
    """
    # 1. Parameter Extraction & Cleaning
    vol_20 = bar_context.get('vol_20', 0.01)
    atr_14 = bar_context.get('atr_14', 0.003)
    obi_tau5 = bar_context.get('obi_tau5', 0.0)
    rsi_14 = bar_context.get('rsi_14', 50.0)
    trend_strength = bar_context.get('trend_strength', 0.0)
    is_us_session = bar_context.get('is_us_session', False)
    funding_rate = bar_context.get('funding_rate', 0.0)
    autocorr_5 = bar_context.get('autocorr_5', 0.0)

    # 2. Threshold & Filter Logic
    spread_bps = bar_context.get('spread_bps', 1.0)
    base_thresh = 0.0019

    # Adjust thresholds based on spread and funding costs
    long_thresh = base_thresh + (spread_bps * 0.00005) + max(0, funding_rate * 0.5)
    short_thresh = -base_thresh - (spread_bps * 0.00005) + min(0, funding_rate * 0.5)

    # 3. Microstructure & Indicator Filters
    # Relaxed filters to increase trade frequency while maintaining quality
    obi_confirm_long = obi_tau5 > -0.2
    obi_confirm_short = obi_tau5 < 0.2
    rsi_safe_long = rsi_14 < 75
    rsi_safe_short = rsi_14 > 25

    # 4. Signal Generation
    signal = 0
    confidence = 0.0

    if predicted_return > long_thresh and obi_confirm_long and rsi_safe_long and spread_bps < 7.0:
        signal = 1
        confidence = min(1.2, predicted_return / base_thresh)
    elif predicted_return < short_thresh and obi_confirm_short and rsi_safe_short and spread_bps < 7.0:
        signal = -1
        confidence = min(1.2, abs(predicted_return) / base_thresh)

    # 5. Position Sizing
    if signal != 0:
        # Base 11% size, scaled by confidence
        position_size = 0.11 * confidence
        # Mild reduction in extreme volatility to protect capital
        if vol_20 > 0.015:
            position_size *= 0.85
    else:
        position_size = 0.0

    # 6. Optimized Risk/Reward
    # Fixed TP/SL inspired by the seed's robust performance; avoids widening in vol
    take_profit = 0.0042
    stop_loss = 0.0030

    # 7. Dynamic Hold Time (max_bars)
    if abs(trend_strength) > 0.4:
        max_bars = 5
    else:
        max_bars = 4

    return {
        "signal": signal,
        "position_size": float(position_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
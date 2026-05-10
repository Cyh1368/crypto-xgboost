import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    High-frequency signal generator balancing model predictions with microstructure
    safety filters and session-specific thresholds.
    """
    # 1. Context Extraction
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    obi_tau5 = bar_context.get('obi_tau5', 0.0)
    spread_bps = bar_context.get('spread_bps', 1.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    atr = bar_context.get('atr_14', 0.002)
    close = bar_context.get('close', 1.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Entry Thresholds
    # Threshold lowered to increase volume closer to the high-performing baseline.
    base_thresh = 0.0018

    signal = 0
    # 3. Filtering Strategy
    # Relaxed filters to recapture alpha while maintaining floor protection against toxicity.
    if predicted_return > base_thresh:
        if obi_tau5 > -0.7 and vwap_dev < 0.025 and spread_bps < 10.0:
            if funding < 0.0015:
                signal = 1
    elif predicted_return < -base_thresh:
        if obi_tau5 < 0.7 and vwap_dev > -0.025 and spread_bps < 10.0:
            if funding > -0.0015:
                signal = -1

    # 4. Position Sizing
    # Size increased to capitalize on the high win rate in this regime.
    position_size = 0.13 if signal != 0 else 0.0

    # 5. Risk Management
    # Maintaining seed-based profit targets proved most robust.
    take_profit = 0.004
    stop_loss = 0.003

    # 6. Hold Time
    # Standard 4-bar hold, with an optional extension for trending US sessions.
    max_bars = 5 if (is_us and abs(trend) > 0.4) else 4

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
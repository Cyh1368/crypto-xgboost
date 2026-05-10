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

    # 2. Dynamic Entry Thresholds
    # Reverting to a more robust threshold similar to the seed (0.002)
    # while maintaining a slight session-based adjustment.
    base_thresh = 0.00195
    if is_us:
        base_thresh = 0.00190 # Slightly more aggressive in US
    elif is_asia:
        base_thresh = 0.00210 # More conservative in Asia

    signal = 0
    # 3. Filtering Strategy (The "Safety Switches")
    # Tightened filters to improve win rate and reduce drawdown.
    if predicted_return > base_thresh:
        # Long filter: Require better OBI support and reasonable VWAP deviation
        if obi_tau5 > -0.25 and vwap_dev < 0.009 and spread_bps < 5.0:
            if funding < 0.0004:
                signal = 1
    elif predicted_return < -base_thresh:
        # Short filter: Require better OBI support and reasonable VWAP deviation
        if obi_tau5 < 0.25 and vwap_dev > -0.009 and spread_bps < 5.0:
            if funding > -0.0004:
                signal = -1

    # 4. Position Sizing
    # Stabilized base size at 0.11 to protect Sharpe and Max DD.
    if signal != 0:
        confidence = abs(predicted_return) / 0.002
        position_size = 0.11 * max(0.9, min(1.15, confidence))

        # Volatility Sizing Adjustment: Reduce size if ATR is high relative to price
        atr_pct = (atr / close) if close > 0 else 0.002
        if atr_pct > 0.0075:
            position_size *= 0.85
    else:
        position_size = 0.0

    # 5. Risk Management (Optimized TP/SL)
    # Reverting to Seed's 0.004/0.003 ratio which showed superior stability.
    take_profit = 0.0040
    stop_loss = 0.0030

    # 6. Adaptive Hold Time
    # Standard exit is 4 bars. Extend only if trend is strongly confirming the signal.
    if (signal * trend) > 0.5:
        max_bars = 5
    else:
        max_bars = 4

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
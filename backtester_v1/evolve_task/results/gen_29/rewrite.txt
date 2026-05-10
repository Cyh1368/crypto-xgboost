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
    # Lower threshold slightly compared to seed to increase trade count, 
    # but adjust for session volatility.
    base_thresh = 0.0018
    if is_us:
        base_thresh = 0.00165 # More vol in US, capture more moves
    elif is_asia:
        base_thresh = 0.0020  # Mean reversion likely, be pickier

    signal = 0
    # 3. Filtering Strategy (The "Safety Switches")
    if predicted_return > base_thresh:
        # Long filter: Avoid entering if OBI is heavily against us or price is too far above VWAP
        if obi_tau5 > -0.6 and vwap_dev < 0.012 and spread_bps < 6.5:
            # Further funding filter: avoid pays too much funding
            if funding < 0.0005:
                signal = 1
                
    elif predicted_return < -base_thresh:
        # Short filter: Avoid entering if OBI is heavily positive or price is too far below VWAP
        if obi_tau5 < 0.6 and vwap_dev > -0.012 and spread_bps < 6.5:
            if funding > -0.0005:
                signal = -1

    # 4. Position Sizing
    # Target a slightly higher base size than seed (0.13) scaled by confidence.
    if signal != 0:
        confidence = abs(predicted_return) / 0.002
        position_size = 0.135 * confidence
        # Cap to prevent over-leverage
        position_size = max(0.08, min(0.18, position_size))
        
        # Volatility Sizing Adjustment: Reduce size if ATR is extremely high relative to price
        atr_pct = (atr / close) if close > 0 else 0.002
        if atr_pct > 0.008:
            position_size *= 0.8
    else:
        position_size = 0.0

    # 5. Risk Management (Optimized TP/SL)
    # Using a robust 1.36 reward-to-risk ratio.
    # Slightly wider than Seed to avoid getting stopped out by spikes.
    take_profit = 0.0045
    stop_loss = 0.0033
    
    # 6. Adaptive Hold Time
    # Standard exit is 4 bars (Seed's sweet spot). 
    # Extend to 5 if trending or strong positive autocorrelation is detected.
    if abs(trend) > 0.6 or (is_us and autocorr > 0.1):
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
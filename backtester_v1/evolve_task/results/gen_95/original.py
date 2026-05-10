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
    pressure = bar_context.get('book_pressure_3', 0.0)
    spread_bps = bar_context.get('spread_bps', 1.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)

    # 2. Dynamic Entry Thresholds
    base_thresh = 0.0018
    if is_us:
        base_thresh = 0.00172
    elif is_asia:
        base_thresh = 0.00205

    # Add liquidity penalty to threshold
    effective_thresh = base_thresh + min(0.0002, kyle_lambda * 25.0)

    # 3. Micro-Alignment and Signal Logic
    micro_score = (obi_tau5 + (pressure / 100.0)) / 2.0
    signal = 0
    if predicted_return > effective_thresh:
        if micro_score > -0.6 and vwap_dev < 0.025 and spread_bps < 10.0:
            if funding < 0.0015:
                signal = 1
    elif predicted_return < -effective_thresh:
        if micro_score < 0.6 and vwap_dev > -0.025 and spread_bps < 10.0:
            if funding > -0.0015:
                signal = -1

    # 4. Adaptive Sizing and Risk Management
    if signal != 0:
        # Confidence-weighted sizing
        confidence = abs(predicted_return) / effective_thresh
        position_size = 0.14 * min(1.25, confidence)
        position_size = max(0.10, min(0.18, position_size))

        # Optimized TP/SL with funding alignment bonus
        take_profit = 0.0042
        stop_loss = 0.0032
        if (signal == 1 and funding < 0) or (signal == -1 and funding > 0):
            take_profit += 0.0004

        # Dynamic Hold Time
        max_bars = 5 if (is_us or abs(trend) > 0.35 or abs(autocorr) > 0.15) else 4
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
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
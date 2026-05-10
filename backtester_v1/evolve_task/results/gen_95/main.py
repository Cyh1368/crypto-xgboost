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
    base_thresh = 0.00178
    if is_us:
        base_thresh = 0.0017
    elif is_asia:
        base_thresh = 0.00208

    # Add liquidity and spread penalty to threshold
    effective_thresh = base_thresh + min(0.00025, kyle_lambda * 30.0) + (spread_bps * 0.000008)

    # 3. Micro-Alignment and Signal Logic
    micro_score = (obi_tau5 + (pressure / 100.0)) / 2.0
    signal = 0
    if predicted_return > effective_thresh:
        if micro_score > -0.55 and vwap_dev < 0.022 and spread_bps < 9.5:
            if funding < 0.0016:
                signal = 1
    elif predicted_return < -effective_thresh:
        if micro_score < 0.55 and vwap_dev > -0.022 and spread_bps < 9.5:
            if funding > -0.0016:
                signal = -1

    # 4. Adaptive Sizing and Risk Management
    if signal != 0:
        # Confidence-weighted sizing anchored higher to exploit high win rate
        confidence = abs(predicted_return) / effective_thresh
        position_size = 0.145 * min(1.2, confidence)
        position_size = max(0.10, min(0.19, position_size))

        # Volatility-adjusted TP/SL targets
        vol_5 = bar_context.get('vol_5', 0.002)
        vol_20 = bar_context.get('vol_20', 0.002)
        vol_ratio = (vol_5 / vol_20) if vol_20 > 0 else 1.0
        vol_scale = max(0.88, min(1.15, vol_ratio))

        take_profit = 0.00425 * vol_scale
        stop_loss = 0.0032 * vol_scale

        # Funding alignment bonus for TP
        if (signal == 1 and funding < 0) or (signal == -1 and funding > 0):
            take_profit += 0.00045

        # 5. Dynamic Hold Time
        # Faster exit in non-trending or mean-reverting Asia environments
        if is_us or abs(trend) > 0.4 or abs(autocorr) > 0.18:
            max_bars = 5
        elif is_asia and abs(autocorr) < 0.05:
            max_bars = 3
        else:
            max_bars = 4
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
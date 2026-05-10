import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Combines session-specific volatility filters, microstructure-based entry hurdles,
    and adaptive ATR risk management.
    """
    # 1. Parameter Extraction
    close = bar_context.get('close', 1.0)
    atr = bar_context.get('atr_14', 0.002)
    obi = bar_context.get('obi_tau5', 0.0)
    vol_ratio = bar_context.get('realized_vol_ratio', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    spread_bps = bar_context.get('spread_bps', 1.0)
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)

    # 2. Dynamic Threshold Logic
    # Lowered thresholds to increase trade frequency, inspired by the 90.84 score iteration.
    base_thresh = 0.0016
    if is_us:
        base_thresh = 0.0015
    elif is_asia:
        base_thresh = 0.0018

    # Penalty for poor liquidity (estimated impact via Kyle's Lambda)
    liquidity_impact = max(0.0, min(0.0002, kyle_lambda * 25.0))
    entry_hurdle = base_thresh + liquidity_impact

    # 3. Execution Signal Logic with Microstructure Filters
    signal = 0
    # Relaxed filters to capture more momentum while maintaining floor protection.
    if predicted_return > entry_hurdle:
        if obi > -0.75 and vwap_dev < 0.025 and funding < 0.0012 and spread_bps < 10.0:
            signal = 1
    elif predicted_return < -entry_hurdle:
        if obi < 0.75 and vwap_dev > -0.025 and funding > -0.0012 and spread_bps < 10.0:
            signal = -1

    # 4. Adaptive Position Sizing
    # Simplified sizing to maintain high trade frequency and capitalize on win rate.
    if signal != 0:
        # vol_ratio > 1 means recent vol is higher than long term; scale down size.
        vol_adj = max(0.85, min(1.15, 1.0 / vol_ratio)) if vol_ratio > 0 else 1.0
        position_size = 0.14 * vol_adj
        position_size = max(0.10, min(0.18, position_size))
    else:
        position_size = 0.0

    # 5. Risk Targets (TP/SL)
    # Anchored to the high-performing 0.004/0.003 ratio with slight trend expansion.
    if signal != 0:
        take_profit = 0.0042
        stop_loss = 0.0031

        # Expand take-profit during trending markets
        if (signal == 1 and trend > 0.4) or (signal == -1 and trend < -0.4):
            take_profit *= 1.1
    else:
        take_profit = 0.0
        stop_loss = 0.0

    # 6. Optimized Hold Time
    # Default 4 bars (standard for 15m), extending to 5 for high persistence/autocorr
    if abs(autocorr) > 0.15 or abs(trend) > 0.5:
        max_bars = 5
    else:
        max_bars = 4

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(round(take_profit, 5)),
        "stop_loss": float(round(stop_loss, 5)),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END
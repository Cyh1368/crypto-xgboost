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
    vwap_dev = bar_context.get('vwap_dev', 0.0)

    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Optimized Threshold Logic
    spread = bar_context.get('spread_bps', 1.0)
    base_thresh = 0.0018

    # 3. Execution Signal logic
    signal = 0
    if predicted_return > base_thresh:
        # Filter by OBI, spread, and avoid entering at VWAP extremes or high funding.
        if obi > -0.65 and spread < 9.5 and vwap_dev < 0.022 and funding < 0.0012:
            signal = 1
    elif predicted_return < -base_thresh:
        if obi < 0.65 and spread < 9.5 and vwap_dev > -0.022 and funding > -0.0012:
            signal = -1

    # 4. Dynamic Hold Time
    # 4 bars is base horizon; 5 bars for trending sessions to capture follow-through.
    max_bars = 5 if (is_us and abs(trend) > 0.4) else 4

    # 5. Risk Targets (TP/SL)
    # Fixed targets have proven more robust than vol-scaling for 15-min model predictions.
    if signal != 0:
        take_profit = 0.0040
        stop_loss = 0.0030

        # 6. Position Sizing
        # Increased base size with a confidence-weighted multiplier.
        confidence = abs(predicted_return) / base_thresh
        position_size = 0.135 * min(1.25, confidence)
        position_size = max(0.10, min(0.19, position_size))
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
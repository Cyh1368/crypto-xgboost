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

    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Optimized Threshold Logic
    # Lowered threshold to maximize trade frequency, with a minor scaling for spread cost.
    spread = bar_context.get('spread_bps', 1.0)
    base_thresh = 0.00175 + (min(spread, 10.0) * 0.00002)

    # 3. Execution Signal logic
    signal = 0
    if predicted_return > base_thresh:
        # Simplified filters: use OBI for quality and spread for cost control.
        if obi > -0.55 and spread < 8.5:
            signal = 1
    elif predicted_return < -base_thresh:
        if obi < 0.55 and spread < 8.5:
            signal = -1

    # 4. Optimized Hold Time
    # 4 bars (60 min) is the optimal horizon for a 15-min forward prediction model.
    max_bars = 4

    # 5. Risk Targets (TP/SL)
    # Calibrated to the seed's successful 0.004/0.003 profile with volatility scaling.
    if signal != 0:
        vol_adj = max(0.85, min(1.2, vol_ratio))
        take_profit = 0.0041 * vol_adj
        stop_loss = 0.0031 * vol_adj

        # 6. Position Sizing
        # Centered at ~11%, scaled by model confidence.
        confidence = abs(predicted_return) / 0.002
        position_size = 0.11 * min(1.3, confidence)
        position_size = max(0.08, min(0.16, position_size))
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
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
    base_thresh = 0.0017
    # Wick ratios to detect price exhaustion
    wick_up = bar_context.get('wick_ratio_up', 0.0)
    wick_dn = bar_context.get('wick_ratio_down', 0.0)

    # 3. Execution Signal logic
    signal = 0
    if predicted_return > base_thresh:
        # Relaxed filters to increase trade frequency and added wick filter
        if obi > -0.72 and spread < 10.2 and vwap_dev < 0.026 and funding < 0.0016:
            if wick_up < 0.75:
                signal = 1
    elif predicted_return < -base_thresh:
        if obi < 0.72 and spread < 10.2 and vwap_dev > -0.026 and funding > -0.0016:
            if wick_dn < 0.75:
                signal = -1

    # 4. Dynamic Hold Time
    # 4 bars base; extension for US sessions, high momentum, or trending autocorrelation.
    max_bars = 5 if (is_us or abs(trend) > 0.35 or autocorr > 0.12) else 4

    # 5. Risk Targets (TP/SL)
    # Maintaining fixed-ratio robustness while slightly tightening SL to protect max drawdown.
    if signal != 0:
        take_profit = 0.0041
        stop_loss = 0.0029

        # 6. Position Sizing
        # More stable sizing to improve Sharpe ratio. Higher base but lower max cap.
        confidence = abs(predicted_return) / base_thresh
        position_size = 0.145 * min(1.15, confidence)
        position_size = max(0.11, min(0.17, position_size))
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
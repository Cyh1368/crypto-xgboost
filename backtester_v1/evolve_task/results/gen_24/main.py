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

    # 2. Liquidity-Adjusted Dynamic Threshold
    # Lowered base threshold to increase trade frequency, with a scaled liquidity penalty.
    liquidity_penalty = max(0.0, min(0.0004, kyle_lambda * 40.0))
    base_thresh = 0.0017 + liquidity_penalty

    # 3. Execution Signal Logic
    signal = 0
    if predicted_return > base_thresh:
        # Relaxed filters to capture more opportunities while using OBI for quality
        if rsi < 82 and funding < 0.0008 and obi > -0.45:
            signal = 1
    elif predicted_return < -base_thresh:
        if rsi > 18 and funding > -0.0008 and obi < 0.45:
            signal = -1

    # 4. Adaptive Hold Time (Max Bars)
    # Standardize on 4-5 bars; US session and strong trends favor longer holds.
    max_bars = 5 if (abs(trend) > 0.5 or (is_us and autocorr > 0.0)) else 4

    # 5. Dynamic Risk Targets (TP/SL)
    # Anchored to the successful 0.004/0.003 ratio (1.33 R/R)
    if signal != 0:
        atr_pct = (atr / close) if close > 0 else 0.003

        # Asymmetry: In trending markets, expand targets slightly
        if (signal == 1 and trend > 0.3) or (signal == -1 and trend < -0.3):
            tp_mult, sl_mult = 1.45, 1.05
        else:
            tp_mult, sl_mult = 1.4, 1.05

        take_profit = max(0.0039, min(0.014, atr_pct * tp_mult))
        stop_loss = max(0.0030, min(0.009, atr_pct * sl_mult))

        # 6. Position Sizing
        # Increased base size and adjusted confidence scaling.
        confidence = abs(predicted_return) / 0.0018
        vol_scale = max(0.85, min(1.25, vol_ratio))

        position_size = 0.11 * confidence * vol_scale
        position_size = max(0.07, min(0.17, position_size))
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
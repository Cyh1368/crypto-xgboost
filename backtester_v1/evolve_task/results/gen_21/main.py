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

    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Dynamic Threshold Logic
    # Lowered threshold to increase trade frequency, using OBI as a lead filter.
    # Base threshold anchored near the high-performing seed value (0.002).
    obi = bar_context.get('obi_tau5', 0.0)
    base_thresh = 0.0018

    # 3. Execution Signal Logic
    signal = 0
    if predicted_return > base_thresh:
        # Loosened filters: only block if RSI is extremely high or book is heavily opposed.
        if rsi < 82 and funding < 0.0007 and obi > -0.45:
            signal = 1
    elif predicted_return < -base_thresh:
        if rsi > 18 and funding > -0.0007 and obi < 0.45:
            signal = -1

    # 4. Adaptive Hold Time (Max Bars)
    # Autocorrelation gauges if we are in a trending or mean-reverting regime.
    if is_us:
        # US Session generally trends more
        max_bars = 5 if autocorr > 0.0 else 4
    elif is_asia:
        # Asia Session often mean-reverts
        max_bars = 4 if autocorr > 0.1 else 3
    else:
        max_bars = 4

    # Increase hold if trend strength is significant
    if abs(trend) > 0.5:
        max_bars += 1

    # 5. Dynamic Risk Targets (TP/SL)
    # Anchored to the successful 0.004/0.003 ratio (1.33 R/R)
    # Adjusted slightly by ATR and trend alignment.
    if signal != 0:
        # Base TP/SL relative to ATR or fixed baseline
        atr_pct = (atr / close) if close > 0 else 0.003

        # Asymmetry: Expand TP and hold slightly tighter SL in trending markets
        if (signal == 1 and trend > 0.4) or (signal == -1 and trend < -0.4):
            tp_mult, sl_mult = 1.6, 1.0
        else:
            tp_mult, sl_mult = 1.4, 1.0

        # Target TP/SL floor near 0.004 / 0.003 while allowing expansion for volatility
        take_profit = max(0.0040, min(0.016, atr_pct * tp_mult * 1.1))
        stop_loss = max(0.0030, min(0.012, atr_pct * sl_mult * 1.1))

        # 6. Position Sizing
        # Increased base size, scaled by prediction confidence and volatility expansion (vol_ratio)
        confidence = abs(predicted_return) / 0.0018
        vol_scale = max(0.9, min(1.2, vol_ratio))

        position_size = 0.12 * confidence * vol_scale
        position_size = max(0.06, min(0.18, position_size))
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
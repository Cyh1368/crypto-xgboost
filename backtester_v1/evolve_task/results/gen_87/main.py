import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Adaptive microstructure-driven signal generation with conviction-based sizing
    and volatility-elastic exit management. Combines high-precision thresholding
    with proven microstructure filters and dynamic position scaling.
    """
    import math
    import numpy as np

    def _get(key, default=0.0):
        v = bar_context.get(key, default)
        return default if v is None else v

    # 1. CONTEXT EXTRACTION & NORMALIZATION
    # Volatility Regimes and Price Action
    v5 = max(float(_get("vol_5", 0.015)), 1e-8)
    v20 = max(float(_get("vol_20", 0.015)), 1e-8)
    v60 = max(float(_get("vol_60", 0.015)), 1e-8)
    vol_impulse = v5 / v20  # Current vs recent volatility (0.5-2.0 range)
    vol_trend = v5 / v60    # Current vs long-term volatility
    trend_strength = float(_get("trend_strength", 0.0))
    vwap_dev = float(_get("vwap_dev", 0.0))

    # Microstructure Factors: Refined weights including longer horizon OBI
    obi1 = float(_get("obi_tau1"))
    obi3 = float(_get("obi_tau3"))
    obi5 = float(_get("obi_tau5"))
    obi10 = float(_get("obi_tau10"))
    book_pressure = float(_get("book_pressure_3"))

    micro_quality = (0.28 * obi1 + 0.22 * obi3 + 0.20 * obi5 + 0.15 * obi10 + 0.15 * book_pressure)

    # Execution Costs & Carry
    spread_bps = float(_get("spread_bps"))
    funding = float(_get("funding_rate"))
    is_us = bool(_get("is_us_session", False))

    # 2. TREND-ADAPTIVE ELASTIC THRESHOLD
    threshold_base = 0.00171
    # Multiplier scales for volatility impulse and trend momentum
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_impulse))
    trend_adj = 1.0 - (0.045 * min(1.2, trend_strength)) # Lower threshold in trending mkts
    current_threshold = threshold_base * elastic_multiplier * trend_adj

    # 3. SIGNAL GENERATION WITH MEAN-REVERSION & QUALITY FILTERS
    raw_signal = 0
    signal_conviction = 0.0

    if predicted_return > current_threshold:
        # Guard: Avoid long if excessively above VWAP or poor microstructure
        if micro_quality > -0.82 and funding < 0.0028 and spread_bps < 10.2 and vwap_dev < 0.0075:
            raw_signal = 1
            signal_conviction = predicted_return / current_threshold
    elif predicted_return < -current_threshold:
        # Guard: Avoid short if excessively below VWAP or poor microstructure
        if micro_quality < 0.82 and funding > -0.0028 and spread_bps < 10.2 and vwap_dev > -0.0075:
            raw_signal = -1
            signal_conviction = abs(predicted_return) / current_threshold

    # 4. CONVICTION-BASED POSITION SIZING
    if raw_signal == 0:
        position_size = 0.0
    else:
        # Base size calibrated for high win-rate leverage
        base_size = 0.145

        # Continuous conviction bonus: rewards high-conviction signals proportionally
        # Conviction ratio > 1.0 means signal exceeds threshold
        # Bonus scales from 0 to 0.035 as conviction increases
        conviction_bonus = min(0.035, max(0, (signal_conviction - 1.0) * 0.06))

        # Session liquidity bonus
        session_bonus = 0.01 if is_us else 0.0

        # Funding adjustment: Mild penalty for adverse carry
        funding_filter = 1.0
        if (raw_signal == 1 and funding > 0.0015) or (raw_signal == -1 and funding < -0.0015):
            funding_filter = 0.90

        # Composite position size with bounds
        position_size = (base_size + conviction_bonus + session_bonus) * funding_filter
        position_size = max(0.10, min(0.18, position_size))

    # 5. VOLATILITY-ELASTIC EXIT MANAGEMENT
    # Take-profit elasticity: Expands in high vol_trend (sustained moves)
    # Range: 0.00376 (low vol) to 0.00424 (high vol)
    tp_elasticity = 0.94 + (0.06 * min(2.0, vol_trend))
    take_profit = 0.0040 * tp_elasticity

    # Stop-loss elasticity: Tighter in high vol_impulse (volatile bars)
    # Range: 0.00291 (high vol) to 0.00309 (low vol)
    sl_elasticity = 0.97 + (0.03 * min(2.0, vol_impulse))
    stop_loss = 0.0030 * sl_elasticity

    # 6. ADAPTIVE TIME HORIZON
    # Exit faster if long-term volatility is declining (stagnation risk)
    max_bars = 4
    if vol_trend < 0.8:
        max_bars = 3

    return {
        "signal":        int(raw_signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
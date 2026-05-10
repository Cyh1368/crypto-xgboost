import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    A layered architectural approach that separates context normalization,
    quality scoring, and volatility-elastic execution.
    """
    import math
    import numpy as np

    # 1. CONTEXT NORMALIZATION LAYER
    def _fetch(key, default=0.0):
        v = bar_context.get(key, default)
        return default if v is None else float(v)

    # Volatility Regimes
    v5 = max(_fetch("vol_5", 0.015), 1e-8)
    v20 = max(_fetch("vol_20", 0.015), 1e-8)
    v60 = max(_fetch("vol_60", 0.015), 1e-8)
    vol_impulse = v5 / v20 # Current vs recent volatility
    vol_trend = v5 / v60    # Current vs long-term volatility

    # Microstructure Factors: Refined weighting from high-performance seeds
    micro_quality = (0.32 * _fetch("obi_tau1") +
                     0.24 * _fetch("obi_tau3") +
                     0.24 * _fetch("obi_tau5") +
                     0.20 * _fetch("book_pressure_3"))
    spread = _fetch("spread_bps")

    # Session Context
    is_us = bool(bar_context.get("is_us_session", False))
    funding = _fetch("funding_rate")

    # 2. SIGNAL SCORING LAYER
    # Determine the "Elastic Threshold": Lowered base to increase trade frequency
    threshold_base = 0.00172
    # Scaling factor: range 0.88 to 1.24 depending on volatility impulse
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_impulse))
    current_threshold = threshold_base * elastic_multiplier

    raw_signal = 0
    signal_conviction = 0.0

    if predicted_return > current_threshold:
        # Microstructure filter: Loosened to capture more valid momentum moves
        if micro_quality > -0.85 and funding < 0.0025 and spread < 10.5:
            raw_signal = 1
            signal_conviction = abs(predicted_return) / current_threshold
    elif predicted_return < -current_threshold:
        if micro_quality < 0.85 and funding > -0.0025 and spread < 10.5:
            raw_signal = -1
            signal_conviction = abs(predicted_return) / current_threshold

    # 3. EXECUTION LOGIC LAYER
    # Position sizing: Rewarding win rate with slightly larger base size
    if raw_signal == 0:
        pos_size = 0.0
    else:
        # Scale sizing by conviction and session liquidity without harsh penalties
        base_size = 0.145
        conviction_bonus = min(0.035, max(0, (signal_conviction - 1.0) * 0.06))
        session_bonus = 0.01 if is_us else 0.0

        # Funding Adjustment: Mild reduction for negative carry trades
        funding_filter = 1.0
        if (raw_signal == 1 and funding > 0.0015) or (raw_signal == -1 and funding < -0.0015):
            funding_filter = 0.90

        pos_size = (base_size + conviction_bonus + session_bonus) * funding_filter
        pos_size = max(0.10, min(0.18, pos_size))

    # Exit parameters: Anchoring to the robust 0.004 TP / 0.003 SL
    # Dampened elasticity to maintain win-rate stability
    tp_elasticity = 0.94 + (0.06 * min(2.0, vol_trend))
    sl_elasticity = 0.97 + (0.03 * min(2.0, vol_impulse))

    take_profit = 0.0040 * tp_elasticity
    stop_loss = 0.0030 * sl_elasticity

    # Time-based risk
    # Exit faster if volatility is trending down (stagnation risk)
    max_bars = 4
    if vol_trend < 0.8:
        max_bars = 3

    return {
        "signal": int(raw_signal),
        "position_size": float(pos_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Combines high-precision thresholding with microstructure filters and
    volatility-adjusted exit logic.
    """
    import math
    import numpy as np

    def _get(key, default=0.0):
        v = bar_context.get(key, default)
        return default if v is None else v

    # 1. Context & Regime
    v5 = max(float(_get("vol_5", 0.015)), 1e-8)
    v20 = max(float(_get("vol_20", 0.015)), 1e-8)
    v60 = max(float(_get("vol_60", 0.015)), 1e-8)
    vol_impulse = v5 / v20
    vol_trend = v5 / v60

    micro_quality = (0.32 * float(_get("obi_tau1")) +
                     0.24 * float(_get("obi_tau3")) +
                     0.24 * float(_get("obi_tau5")) +
                     0.20 * float(_get("book_pressure_3")))

    spread_bps = float(_get("spread_bps"))
    funding = float(_get("funding_rate"))
    is_us = bool(_get("is_us_session", False))

    # Price action confirmations
    rsi_14 = float(_get("rsi_14", _get("rsi", 50.0)))
    bb_pct = float(_get("bb_pct", 0.5))

    # 2. Signal Generation
    threshold_base = 0.00171
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_impulse))
    current_threshold = threshold_base * elastic_multiplier

    signal = 0
    conviction_ratio = 0.0
    if predicted_return > current_threshold:
        # Long: require positive microstructure, reasonable funding/spread, AND price action confirmation
        if (micro_quality > -0.85 and funding < 0.0025 and spread_bps < 10.5 and
            rsi_14 < 70.0 and bb_pct < 0.8):
            signal = 1
            conviction_ratio = predicted_return / current_threshold
    elif predicted_return < -current_threshold:
        # Short: require negative microstructure, reasonable funding/spread, AND price action confirmation
        if (micro_quality < 0.85 and funding > -0.0025 and spread_bps < 10.5 and
            rsi_14 > 30.0 and bb_pct > 0.2):
            signal = -1
            conviction_ratio = abs(predicted_return) / current_threshold

    # 3. Position Sizing
    if signal == 0:
        position_size = 0.0
    else:
        base_size = 0.145

        # Conviction-scaled bonus with quadratic component for strong signals
        if conviction_ratio > 1.5:
            # High conviction: scale quadratically for aggressive sizing
            conviction_bonus = min(0.040, (conviction_ratio - 1.0) * 0.055 + 0.005 * max(0, conviction_ratio - 1.5) ** 1.3)
        else:
            # Standard linear scaling for moderate signals
            conviction_bonus = min(0.035, max(0, (conviction_ratio - 1.0) * 0.06))

        session_bonus = 0.01 if is_us else 0.0

        funding_filter = 1.0
        if (signal == 1 and funding > 0.0010) or (signal == -1 and funding < -0.0010):
            funding_filter = 0.85

        position_size = max(0.10, min(0.18, (base_size + conviction_bonus + session_bonus) * funding_filter))

    # 4. Volatility-Elastic Exits
    tp_elasticity = 0.94 + (0.06 * min(2.0, vol_trend))
    sl_elasticity = 0.97 + (0.03 * min(2.0, vol_impulse))
    take_profit = 0.0040 * tp_elasticity
    stop_loss = 0.0030 * sl_elasticity

    # 5. Time-based Exit
    max_bars = 3 if vol_trend < 0.8 else 4

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
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
    vol_impulse = max(0.5, min(2.0, v5 / v20))
    vol_trend = max(0.5, min(2.0, v5 / v60))

    obi1 = float(_get("obi_tau1"))
    obi3 = float(_get("obi_tau3"))
    obi5 = float(_get("obi_tau5"))
    obi10 = float(_get("obi_tau10"))
    book_pressure = float(_get("book_pressure_3"))
    depth_ratio_5 = max(float(_get("depth_ratio_5", 1.0)), 1e-8)
    depth_ratio_10 = max(float(_get("depth_ratio_10", 1.0)), 1e-8)
    spread_bps = float(_get("spread_bps"))
    kyle_lambda = abs(float(_get("kyle_lambda_est")))
    vwap_dev = float(_get("vwap_dev"))
    bb_pct = float(_get("bb_pct", 0.5))
    rsi6 = float(_get("rsi_6", _get("rsi", 50.0)))
    rsi14 = float(_get("rsi_14", _get("rsi", 50.0)))
    macd_signal = float(_get("macd_signal"))
    trend_strength = float(_get("trend_strength"))
    ret1 = float(_get("ret_1"))
    ret3 = float(_get("ret_3"))
    ret6 = float(_get("ret_6"))
    funding = float(_get("funding_rate"))
    funding_ma = float(_get("funding_8h_ma"))
    minutes_to_funding = float(_get("minutes_to_funding", 999.0))
    is_us = bool(_get("is_us_session", False))
    is_asia = bool(_get("is_asia_session", False))
    is_weekend = bool(_get("is_weekend", False))

    micro_quality = (0.28 * obi1 +
                     0.22 * obi3 +
                     0.22 * obi5 +
                     0.10 * obi10 +
                     0.18 * book_pressure)

    liquidity_quality = (0.55 * min(1.5, depth_ratio_5) +
                         0.45 * min(1.5, depth_ratio_10)) - 0.06 * spread_bps - 0.9 * min(1.0, kyle_lambda)
    session_bias = 0.06 if is_us else (-0.03 if is_asia else 0.0)
    weekend_penalty = 0.04 if is_weekend else 0.0

    # 2. Signal Generation
    threshold_base = 0.00171
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_impulse))
    current_threshold = threshold_base * elastic_multiplier

    signal = 0
    conviction_ratio = 0.0

    trend_align_long = (trend_strength > 0 and macd_signal >= 0 and rsi6 >= 48 and rsi14 >= 50 and ret3 > -0.001)
    trend_align_short = (trend_strength < 0 and macd_signal <= 0 and rsi6 <= 52 and rsi14 <= 50 and ret3 < 0.001)
    mean_revert_long = (bb_pct < 0.25 and vwap_dev < -0.001 and ret1 > -0.002)
    mean_revert_short = (bb_pct > 0.75 and vwap_dev > 0.001 and ret1 < 0.002)

    # Funding asymmetry: avoid paying into crowded longs near funding, but allow shorts more freely
    long_funding_ok = (funding <= funding_ma + 0.00025) and (minutes_to_funding > 12.0 or funding <= 0.0015)
    short_funding_ok = (funding >= funding_ma - 0.00025) and (minutes_to_funding > 12.0 or funding >= -0.0015)

    if predicted_return > current_threshold:
        if micro_quality > -0.70 and liquidity_quality > -0.20 and spread_bps < 11.0 and long_funding_ok:
            if trend_align_long or mean_revert_long or session_bias > 0.0:
                signal = 1
                conviction_ratio = predicted_return / current_threshold
    elif predicted_return < -current_threshold:
        if micro_quality < 0.70 and liquidity_quality > -0.20 and spread_bps < 11.0 and short_funding_ok:
            if trend_align_short or mean_revert_short or session_bias <= 0.0:
                signal = -1
                conviction_ratio = abs(predicted_return) / current_threshold

    # 3. Position Sizing
    if signal == 0:
        position_size = 0.0
    else:
        base_size = 0.142
        conviction_bonus = min(0.040, max(0, (conviction_ratio - 1.0) * 0.065))

        # stronger sizing only when book/liquidity agree with the model direction
        micro_bonus = 0.012 if ((signal == 1 and micro_quality > 0) or (signal == -1 and micro_quality < 0)) else 0.0
        liquidity_bonus = 0.010 if liquidity_quality > 0.15 else 0.0
        session_bonus = 0.012 if is_us else (0.004 if is_asia else 0.0)

        funding_filter = 1.0
        if (signal == 1 and funding > 0.0012) or (signal == -1 and funding < -0.0012):
            funding_filter = 0.88

        if signal == 1 and (bb_pct > 0.70 or vwap_dev > 0.002):
            base_size -= 0.006
        if signal == -1 and (bb_pct < 0.30 or vwap_dev < -0.002):
            base_size -= 0.006

        position_size = (base_size + conviction_bonus + micro_bonus + liquidity_bonus + session_bonus) * funding_filter
        position_size = max(0.08, min(0.19, position_size))

    # 4. Volatility-Elastic Exits
    tp_elasticity = 0.93 + (0.07 * min(2.0, vol_trend))
    sl_elasticity = 0.96 + (0.04 * min(2.0, vol_impulse))

    # widen TP slightly when trend is strong, tighten when spread/liquidity deteriorate
    tp_adjust = 1.0 + (0.04 if abs(trend_strength) > 0.45 else 0.0) - (0.03 if spread_bps > 9.5 else 0.0)
    sl_adjust = 1.0 + (0.03 if kyle_lambda > 0.8 else 0.0) + (0.02 if spread_bps > 9.5 else 0.0)

    take_profit = 0.0040 * tp_elasticity * tp_adjust
    stop_loss = 0.0030 * sl_elasticity * sl_adjust

    # 5. Time-based Exit
    max_bars = 4
    if vol_trend < 0.8 or is_weekend:
        max_bars = 3
    if abs(trend_strength) > 0.45 and signal != 0:
        max_bars = 5

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
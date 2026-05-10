import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Combines high-precision volatility-adaptive thresholding with
    proven microstructure filters and high-conviction sizing.
    """
    import math
    import numpy as np

    def _get(key, default=0.0):
        v = bar_context.get(key, default)
        return default if v is None else v

    # 1. Microstructure and Context Extraction
    obi1 = float(_get("obi_tau1"))
    obi3 = float(_get("obi_tau3"))
    obi5 = float(_get("obi_tau5"))
    obi10 = float(_get("obi_tau10"))
    book_pressure = float(_get("book_pressure_3"))
    spread_bps = float(_get("spread_bps"))
    vwap_dev = float(_get("vwap_dev"))
    depth_ratio_5 = float(_get("depth_ratio_5", 1.0))
    kyle_lambda = abs(float(_get("kyle_lambda_est", 0.0)))

    # Price Action & Volatility
    vol5 = max(float(_get("vol_5", 0.015)), 1e-8)
    vol20 = max(float(_get("vol_20", 0.015)), 1e-8)
    vol60 = max(float(_get("vol_60", 0.015)), 1e-8)
    atr14 = max(float(_get("atr_14", _get("atr", 0.001))), 1e-8)
    vol_regime = vol5 / vol20
    vol_trend = vol5 / vol60
    rsi14 = float(_get("rsi_14", _get("rsi", 50.0)))
    macd_signal = float(_get("macd_signal", 0.0))
    bb_pct = float(_get("bb_pct", 0.5))
    trend_strength = float(_get("trend_strength", 0.0))
    autocorr5 = float(_get("autocorr_5", 0.0))
    ret1 = float(_get("ret_1", 0.0))
    ret3 = float(_get("ret_3", 0.0))
    ret6 = float(_get("ret_6", 0.0))

    # Macro & Session
    funding = float(_get("funding_rate"))
    funding_ma = float(_get("funding_8h_ma", 0.0))
    minutes_to_funding = float(_get("minutes_to_funding", 999.0))
    is_us = bool(_get("is_us_session", False))
    is_asia = bool(_get("is_asia_session", False))
    is_weekend = bool(_get("is_weekend", False))

    # 2. Regime Quality Metrics
    # micro_quality: Positive means book supports long, negative supports short.
    micro_quality = (0.32 * obi1 + 0.24 * obi3 + 0.24 * obi5 + 0.20 * book_pressure)

    # 3. Dynamic Signal Thresholding
    threshold_base = 0.00172
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_regime))
    current_threshold = threshold_base * elastic_multiplier

    signal = 0
    signal_conviction = 0.0
    # Regime-aware entry filters: keep the base edge but reject low-quality states.
    trend_long = (trend_strength > 0.15 and macd_signal >= -0.0002 and autocorr5 > -0.05)
    trend_short = (trend_strength < -0.15 and macd_signal <= 0.0002 and autocorr5 < 0.05)
    long_exhausted = (rsi14 > 72.0) or (bb_pct > 0.90 and vwap_dev > 0.0025)
    short_exhausted = (rsi14 < 28.0) or (bb_pct < 0.10 and vwap_dev < -0.0025)
    execution_ok = (spread_bps < 10.0 and depth_ratio_5 > 0.85 and kyle_lambda < 0.08)
    carry_ok_long = (funding < 0.0025 and funding >= funding_ma - 0.0004)
    carry_ok_short = (funding > -0.0025 and funding <= funding_ma + 0.0004)
    near_funding = (minutes_to_funding > 18.0)

    if predicted_return > current_threshold:
        if (micro_quality > -0.85 and execution_ok and carry_ok_long and near_funding and
            not long_exhausted and (trend_long or ret3 > 0.0 or ret6 > 0.0)):
            signal = 1
            signal_conviction = abs(predicted_return) / current_threshold
    elif predicted_return < -current_threshold:
        if (micro_quality < 0.85 and execution_ok and carry_ok_short and near_funding and
            not short_exhausted and (trend_short or ret3 < 0.0 or ret6 < 0.0)):
            signal = -1
            signal_conviction = abs(predicted_return) / current_threshold

    # 4. Asymmetric Position Sizing
    if signal == 0:
        position_size = 0.0
    else:
        base_size = 0.145
        conviction_bonus = min(0.035, max(0, (signal_conviction - 1.0) * 0.06))
        session_bonus = 0.01 if is_us else 0.0

        funding_filter = 1.0
        if (signal == 1 and funding > 0.0015) or (signal == -1 and funding < -0.0015):
            funding_filter = 0.90

        position_size = (base_size + conviction_bonus + session_bonus) * funding_filter
        position_size = min(0.18, max(0.10, position_size))

    # 5. Volatility-Elastic Risk Management
    tp_elasticity = 0.94 + (0.06 * min(2.0, vol_trend))
    sl_elasticity = 0.97 + (0.03 * min(2.0, vol_regime))

    # Slightly tighter protection when spread/impact are poor; allow more room when ATR is large.
    friction = 1.0 + min(0.08, 0.01 * max(0.0, spread_bps - 6.0)) + min(0.06, 0.5 * kyle_lambda)
    atr_scale = min(1.12, max(0.88, atr14 / max(vol5, 1e-8)))
    take_profit = 0.0040 * tp_elasticity * atr_scale / friction
    stop_loss = 0.0030 * sl_elasticity * friction * max(0.92, 1.0 / atr_scale)
    max_bars = 4
    if vol_trend < 0.8 or is_weekend or is_asia:
        max_bars = 3

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
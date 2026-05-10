import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Regime-aware signal generator with asymmetric filters and volatility-adjusted exits.
    """
    import math
    import numpy as np

    def _get(key, default=0.0):
        v = bar_context.get(key, default)
        return default if v is None else v

    # --- Microstructure ---
    obi1 = float(_get("obi_tau1"))
    obi3 = float(_get("obi_tau3"))
    obi5 = float(_get("obi_tau5"))
    obi10 = float(_get("obi_tau10"))
    book_pressure = float(_get("book_pressure_3"))
    spread_bps = float(_get("spread_bps"))
    depth5 = float(_get("depth_ratio_5"))
    depth10 = float(_get("depth_ratio_10"))
    kyle_lambda = abs(float(_get("kyle_lambda_est")))

    # --- Price action / momentum ---
    ret1 = float(_get("ret_1"))
    ret3 = float(_get("ret_3"))
    ret6 = float(_get("ret_6"))
    ret12 = float(_get("ret_12"))
    vol5 = max(float(_get("vol_5")), 1e-8)
    vol20 = max(float(_get("vol_20")), 1e-8)
    vol60 = max(float(_get("vol_60")), 1e-8)
    rsi6 = float(_get("rsi_6"))
    rsi14 = float(_get("rsi_14"))
    macd_signal = float(_get("macd_signal"))
    bb_pct = float(_get("bb_pct"))
    atr14 = max(float(_get("atr_14", _get("atr"))), 1e-8)
    momentum = float(_get("momentum"))
    wick_up = float(_get("wick_ratio_up"))
    wick_dn = float(_get("wick_ratio_down"))
    volume_ratio_5 = float(_get("volume_ratio_5"))
    volume_ratio_20 = float(_get("volume_ratio_20"))
    vwap_dev = float(_get("vwap_dev"))
    autocorr5 = float(_get("autocorr_5"))
    skew = float(_get("skew"))
    kurt = float(_get("kurt"))
    trend_strength = float(_get("trend_strength"))

    # --- Macro / time ---
    funding = float(_get("funding_rate"))
    funding_ma = float(_get("funding_8h_ma"))
    is_asia = bool(_get("is_asia_session", False))
    is_us = bool(_get("is_us_session", False))
    is_weekend = bool(_get("is_weekend", False))
    minutes_to_funding = float(_get("minutes_to_funding", 9999.0))

    # Regimes
    vol_regime = max(0.55, min(1.9, vol5 / vol20))
    atr_regime = max(0.7, min(1.6, atr14 / max(vol60, 1e-8)))
    spread_penalty = 1.0 + min(1.25, spread_bps / 12.0)

    # Directional quality components
    micro_long = (
        0.28 * obi1 + 0.18 * obi3 + 0.14 * obi5 + 0.10 * obi10
        + 0.16 * book_pressure + 0.08 * depth5 + 0.06 * depth10
        - 0.10 * kyle_lambda
    )
    price_long = (
        0.20 * ret1 + 0.18 * ret3 + 0.14 * ret6 + 0.10 * ret12
        + 0.14 * momentum + 0.10 * trend_strength + 0.08 * macd_signal
        + 0.08 * (bb_pct - 0.5) + 0.04 * vwap_dev + 0.04 * autocorr5
    )
    exhaustion_short = 0.10 * max(0.0, wick_up - wick_dn) + 0.08 * max(0.0, skew) + 0.05 * max(0.0, kurt - 3.0)

    long_score = micro_long + price_long
    short_score = -(0.28 * obi1 + 0.18 * obi3 + 0.14 * obi5 + 0.10 * obi10
                    + 0.16 * book_pressure + 0.08 * depth5 + 0.06 * depth10
                    - 0.10 * kyle_lambda) - (
                    0.20 * ret1 + 0.18 * ret3 + 0.14 * ret6 + 0.10 * ret12
                    + 0.14 * momentum + 0.10 * trend_strength + 0.08 * macd_signal
                    + 0.08 * (0.5 - bb_pct) - 0.04 * vwap_dev - 0.04 * autocorr5
                )

    # Thresholds: tighter in clean regimes, stricter in toxic flow / high-cost regimes
    base = 0.0017 * vol_regime * (0.95 + 0.15 * atr_regime)
    long_thresh = base * spread_penalty
    short_thresh = -base * spread_penalty

    # Context filters
    funding_drift = funding - funding_ma
    carry_ok_long = (funding <= 0.0015 and funding_drift <= 0.0007) or minutes_to_funding > 25.0
    carry_ok_short = (funding >= -0.0015 and funding_drift >= -0.0007) or minutes_to_funding > 25.0
    liquidity_ok = spread_bps <= 10.5 and kyle_lambda <= 0.018
    trend_ok_long = trend_strength > -0.15 and macd_signal > -0.15 and rsi6 < 78.0 and bb_pct > 0.18
    trend_ok_short = trend_strength < 0.15 and macd_signal < 0.15 and rsi6 > 22.0 and bb_pct < 0.82
    regime_ok = not is_weekend or (abs(predicted_return) > 0.0032 and vol_regime > 0.8)

    signal = 0
    if predicted_return > long_thresh:
        if long_score > 0.15 and carry_ok_long and liquidity_ok and trend_ok_long and regime_ok and exhaustion_short < 0.18:
            signal = 1
    elif predicted_return < short_thresh:
        if short_score > 0.15 and carry_ok_short and liquidity_ok and trend_ok_short and regime_ok and exhaustion_short < 0.18:
            signal = -1

    # Asymmetric sizing: stronger when forecast aligns with book + trend, smaller in expensive/noisy regimes
    if signal == 0:
        position_size = 0.0
        take_profit = 0.004
        stop_loss = 0.003
        max_bars = 4
    else:
        conviction = min(1.0, abs(predicted_return) / 0.005)
        alignment = max(0.0, min(1.0, (abs(long_score) / (abs(long_score) + 1.0))))
        session_boost = 0.015 if is_us else (0.005 if is_asia else 0.0)
        base_size = 0.09 + 0.05 * conviction + 0.03 * alignment + session_boost
        if signal == 1:
            base_size += 0.01 if carry_ok_long else -0.01
        else:
            base_size += 0.01 if carry_ok_short else -0.01
        position_size = min(0.22, max(0.07, base_size))

        # Vol-adjusted exits: wider targets when volatility is elevated and signal is cleaner
        tp_base = 0.0036 + 0.0010 * min(1.0, vol_regime - 0.8) + 0.0008 * conviction
        sl_base = 0.0026 + 0.0008 * min(1.0, vol_regime - 0.8) + 0.0004 * (1.0 - alignment)

        if signal == 1:
            take_profit = tp_base * 1.05
            stop_loss = sl_base * 0.95
        else:
            take_profit = tp_base * 1.00
            stop_loss = sl_base * 0.90

        max_bars = 3 if vol_regime > 1.15 or abs(funding_drift) > 0.0008 else 4
        if is_weekend:
            max_bars = max(2, max_bars - 1)

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
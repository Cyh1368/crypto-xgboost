import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Regime-aware signal generator using asymmetric entry filters,
    microstructure confirmation, and volatility-adjusted exits.
    """
    # 1. Core context
    close = bar_context.get('close', 1.0)
    atr = bar_context.get('atr_14', 0.002)
    vol_5 = bar_context.get('vol_5', 0.001)
    vol_20 = bar_context.get('vol_20', 0.001)
    vol_60 = bar_context.get('vol_60', 0.001)
    rsi_6 = bar_context.get('rsi_6', 50.0)
    rsi_14 = bar_context.get('rsi_14', 50.0)
    bb_pct = bar_context.get('bb_pct', 0.5)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    funding_ma = bar_context.get('funding_rate_ma8h', funding)
    minutes_to_funding = bar_context.get('minutes_to_funding', 999.0)

    obi1 = bar_context.get('obi_tau1', 0.0)
    obi3 = bar_context.get('obi_tau3', 0.0)
    obi5 = bar_context.get('obi_tau5', 0.0)
    obi10 = bar_context.get('obi_tau10', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    depth_ratio_5 = bar_context.get('depth_ratio_5', 1.0)
    depth_ratio_10 = bar_context.get('depth_ratio_10', 1.0)
    book_pressure_3 = bar_context.get('book_pressure_3', 0.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)

    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    is_weekend = bar_context.get('is_weekend', False)

    # 2. Regime and cost-aware entry threshold
    spread_penalty = min(0.00045, max(0.0, spread) * 0.00003)
    impact_penalty = min(0.00035, max(0.0, kyle_lambda) * 18.0)
    vol_regime = vol_5 / max(1e-8, vol_20) if vol_20 > 0 else 1.0
    vol_regime = max(0.7, min(1.6, vol_regime))

    base_thresh = 0.00155 + spread_penalty + impact_penalty
    if is_us:
        base_thresh *= 0.95
    elif is_asia:
        base_thresh *= 1.08
    if is_weekend:
        base_thresh *= 1.05

    # Require more edge when volatility is elevated
    entry_hurdle = base_thresh * (0.92 + 0.10 * (vol_regime - 1.0))
    entry_hurdle = max(0.00125, min(0.00245, entry_hurdle))

    # 3. Asymmetric direction filters
    signal = 0

    long_micro_ok = (
        obi1 > -0.15 and obi3 > -0.10 and obi5 > -0.20 and obi10 > -0.25
        and depth_ratio_5 > 0.92 and depth_ratio_10 > 0.90
        and book_pressure_3 > -0.10
        and spread <= 9.0
    )
    short_micro_ok = (
        obi1 < 0.15 and obi3 < 0.10 and obi5 < 0.20 and obi10 < 0.25
        and depth_ratio_5 > 0.90 and depth_ratio_10 > 0.88
        and book_pressure_3 < 0.10
        and spread <= 9.0
    )

    long_momentum_ok = (
        (rsi_6 < 72.0 and rsi_14 < 68.0)
        or (bb_pct < 0.88 and vwap_dev < 0.018)
        or (trend > 0.25 and autocorr > -0.05)
    )
    short_momentum_ok = (
        (rsi_6 > 28.0 and rsi_14 > 32.0)
        or (bb_pct > 0.12 and vwap_dev > -0.018)
        or (trend < -0.25 and autocorr < 0.05)
    )

    # Funding-aware asymmetry: avoid fading crowded longs too aggressively
    long_funding_ok = (funding <= funding_ma + 0.00015) and (minutes_to_funding > 12 or funding <= 0.0005)
    short_funding_ok = (funding >= funding_ma - 0.00015) or (minutes_to_funding > 12)

    if predicted_return > entry_hurdle:
        if long_micro_ok and long_momentum_ok and long_funding_ok:
            signal = 1
    elif predicted_return < -entry_hurdle:
        # Shorts need slightly stronger confirmation because squeezes can be sharper
        if short_micro_ok and short_momentum_ok and short_funding_ok and not (is_us and trend > 0.45):
            signal = -1

    # 4. Volatility-adjusted sizing
    if signal != 0:
        confidence = abs(predicted_return) / max(entry_hurdle, 1e-6)
        confidence = max(0.8, min(1.6, confidence))

        micro_quality = 1.0
        if signal == 1:
            micro_quality += max(0.0, min(0.10, (obi1 + obi3 + obi5) * 0.03))
            micro_quality += max(0.0, min(0.05, book_pressure_3 * 0.05))
        else:
            micro_quality += max(0.0, min(0.10, (-obi1 - obi3 - obi5) * 0.03))
            micro_quality += max(0.0, min(0.05, (-book_pressure_3) * 0.05))

        vol_scale = 1.0 / max(0.85, min(1.35, vol_regime))
        position_size = 0.105 * confidence * micro_quality * vol_scale
        position_size = max(0.075, min(0.175, position_size))
    else:
        position_size = 0.0

    # 5. Dynamic exits
    if signal != 0:
        atr_pct = atr / max(close, 1e-8)
        atr_pct = max(0.0015, min(0.012, atr_pct))

        # Wider targets in strong trend, tighter in mean-reversion / elevated vol
        trend_boost = 1.0 + max(0.0, min(0.18, abs(trend) * 0.10))
        mean_revert_penalty = 1.0 - max(0.0, min(0.12, abs(bb_pct - 0.5) * 0.12))

        take_profit = max(0.0036, min(0.0062, (0.0038 + 0.18 * atr_pct) * trend_boost))
        stop_loss = max(0.0024, min(0.0048, (0.0028 + 0.14 * atr_pct) * mean_revert_penalty))

        # Keep reward/risk asymmetric but avoid over-widening
        if signal == 1 and trend > 0.35:
            take_profit *= 1.08
        elif signal == -1 and trend < -0.35:
            take_profit *= 1.08

        if vol_regime > 1.2:
            stop_loss *= 1.05
    else:
        take_profit = 0.0
        stop_loss = 0.0

    # 6. Hold time selection
    if signal != 0:
        if abs(trend) > 0.45 or abs(autocorr) > 0.12:
            max_bars = 5
        elif signal == 1 and is_us and rsi_14 > 55.0:
            max_bars = 5
        elif signal == -1 and (is_asia or is_weekend):
            max_bars = 4
        else:
            max_bars = 4
    else:
        max_bars = 0

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(round(take_profit, 5)),
        "stop_loss": float(round(stop_loss, 5)),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END
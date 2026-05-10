import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Context-aware asymmetric signal generator.

    Uses predicted return as the base edge, then refines entries with:
    - microstructure confirmation
    - trend / mean-reversion state
    - volatility-adjusted thresholds
    - funding and session guards
    - asymmetric exits and sizing
    """
    import math
    import numpy as np

    def _safe_get(key, default=0.0, cast_fn=float):
        v = bar_context.get(key, default)
        if v is None:
            return default
        try:
            return cast_fn(v)
        except (ValueError, TypeError):
            return default

    # --- Microstructure ---
    obi1 = _safe_get("obi_tau1", 0.0)
    obi3 = _safe_get("obi_tau3", 0.0)
    obi5 = _safe_get("obi_tau5", 0.0)
    obi10 = _safe_get("obi_tau10", 0.0)
    spread_bps = max(_safe_get("spread_bps", 5.0), 1e-8)
    depth_ratio_5 = _safe_get("depth_ratio_5", 1.0)
    depth_ratio_10 = _safe_get("depth_ratio_10", 1.0)
    book_pressure = _safe_get("book_pressure_3", 0.0)
    kyle_lambda = abs(_safe_get("kyle_lambda_est", 0.0))

    # --- Price action / vol ---
    ret_1 = _safe_get("ret_1", 0.0)
    ret_3 = _safe_get("ret_3", 0.0)
    ret_6 = _safe_get("ret_6", 0.0)
    ret_12 = _safe_get("ret_12", 0.0)
    vol_5 = max(_safe_get("vol_5", 0.015), 1e-8)
    vol_20 = max(_safe_get("vol_20", 0.015), 1e-8)
    vol_60 = max(_safe_get("vol_60", 0.015), 1e-8)
    rsi_6 = _safe_get("rsi_6", _safe_get("rsi_14", 50.0))
    rsi_14 = _safe_get("rsi_14", 50.0)
    macd_signal = _safe_get("macd_signal", 0.0)
    bb_pct = _safe_get("bb_pct", 0.5)
    atr_14 = max(_safe_get("atr_14", _safe_get("atr", 0.001)), 1e-8)
    momentum = _safe_get("momentum", 0.0)
    vwap_dev = _safe_get("vwap_dev", 0.0)
    autocorr_5 = _safe_get("autocorr_5", 0.0)
    skew_20 = _safe_get("skew_20", 0.0)
    kurt_20 = _safe_get("kurt_20", 3.0)
    trend_strength = _safe_get("trend_strength", 0.0)

    # --- Macro / time ---
    funding_rate = _safe_get("funding_rate", 0.0)
    funding_8h_ma = _safe_get("funding_8h_ma", 0.0)
    minutes_to_funding = _safe_get("minutes_to_funding", 999.0)
    is_us_session = bool(_safe_get("is_us_session", False, bool))
    is_asia_session = bool(_safe_get("is_asia_session", False, bool))
    is_weekend = bool(_safe_get("is_weekend", False, bool))

    # --- Derived state ---
    vol_ratio_5_20 = vol_5 / vol_20
    vol_ratio_20_60 = vol_20 / vol_60
    vol_regime = max(0.5, min(2.0, max(vol_ratio_5_20, vol_ratio_20_60)))

    recent_momentum = 0.5 * ret_1 + 0.3 * ret_3 + 0.2 * ret_6
    medium_momentum = 0.4 * ret_6 + 0.6 * ret_12
    churn = abs(autocorr_5) + max(0.0, kurt_20 - 3.0) * 0.08 + abs(skew_20) * 0.04

    # Microstructure quality: positive supports long, negative supports short
    micro_long = (
        0.22 * obi1 + 0.18 * obi3 + 0.22 * obi5 + 0.10 * obi10 +
        0.14 * book_pressure + 0.08 * depth_ratio_5 + 0.06 * depth_ratio_10 -
        0.10 * spread_bps / 10.0 - 0.10 * kyle_lambda * 50.0
    )
    micro_short = -micro_long

    # --- Entry logic ---
    signal = 0

    # Dynamic thresholds: stricter in noisy regimes, looser in clean/trending regimes
    base_long_thresh = 0.0018
    base_short_thresh = -0.0018
    thresh_adj = 1.0 + 0.18 * (vol_regime - 1.0) + 0.12 * churn
    long_thresh = base_long_thresh * thresh_adj
    short_thresh = base_short_thresh * thresh_adj

    trend_bias = (
        0.30 * trend_strength +
        0.18 * max(0.0, medium_momentum / 0.002) +
        0.12 * max(0.0, recent_momentum / 0.0015) +
        0.10 * momentum
    )
    mean_revert_bias = (
        0.22 * max(0.0, 0.25 - bb_pct) +
        0.22 * max(0.0, bb_pct - 0.75) +
        0.18 * max(0.0, (50.0 - rsi_14) / 20.0) +
        0.18 * max(0.0, (rsi_14 - 50.0) / 20.0) +
        0.10 * abs(vwap_dev) / max(atr_14, 1e-8)
    )

    funding_pressure = funding_rate - funding_8h_ma
    near_funding = minutes_to_funding < 20.0
    funding_long_ok = funding_rate <= funding_8h_ma + 0.0002 or funding_rate < 0.0
    funding_short_ok = funding_rate >= funding_8h_ma - 0.0002 or funding_rate > 0.0

    # Long candidate
    long_score = (
        predicted_return +
        0.32 * micro_long +
        0.24 * trend_bias +
        0.14 * max(0.0, -vwap_dev / max(atr_14, 1e-8)) +
        0.10 * max(0.0, (45.0 - rsi_6) / 15.0) +
        0.08 * float(is_us_session) -
        0.08 * float(is_weekend)
    )

    # Short candidate
    short_score = (
        -predicted_return +
        0.32 * micro_short +
        0.24 * trend_bias +
        0.14 * max(0.0, vwap_dev / max(atr_14, 1e-8)) +
        0.10 * max(0.0, (rsi_6 - 55.0) / 15.0) +
        0.08 * float(is_us_session) -
        0.08 * float(is_weekend)
    )

    # Mean-reversion becomes more important when trend is weak and churn is elevated
    trend_is_weak = trend_strength < 0.10 or abs(medium_momentum) < 0.0008
    if trend_is_weak and churn < 0.45:
        long_score += 0.18 * mean_revert_bias
        short_score += 0.18 * mean_revert_bias

    if predicted_return > long_thresh and long_score > long_thresh and micro_long > -0.15 and funding_long_ok and not near_funding:
        signal = 1
    elif predicted_return < short_thresh and short_score > abs(short_thresh) and micro_short > -0.15 and funding_short_ok and not near_funding:
        signal = -1

    # Strong flow can override a slightly weaker model prediction
    if signal == 0 and not near_funding:
        if micro_long > 0.22 and trend_bias > 0.05 and predicted_return > 0.0010 and funding_long_ok and spread_bps < 10.0:
            signal = 1
        elif micro_short > 0.22 and trend_bias > 0.05 and predicted_return < -0.0010 and funding_short_ok and spread_bps < 10.0:
            signal = -1

    # --- Risk management / sizing ---
    if signal == 0:
        position_size = 0.0
        take_profit = 0.004
        stop_loss = 0.003
        max_bars = 4
    else:
        # Base size depends on signal quality and liquidity
        quality = max(0.0, min(1.0, 0.5 + 0.7 * abs(predicted_return) / 0.004 + 0.25 * abs(micro_long) - 0.15 * spread_bps / 10.0))
        liquidity_scalar = max(0.65, min(1.15, 1.0 + 0.12 * (1.0 - vol_regime)))
        funding_scalar = max(0.65, min(1.0, 1.0 - abs(funding_pressure) * 60.0))
        weekend_scalar = 0.85 if is_weekend else 1.0
        session_scalar = 1.05 if is_us_session else (0.95 if is_asia_session else 1.0)

        if signal == 1:
            base_position = 0.11
            tp_mult = 1.05 + 0.10 * max(0.0, trend_strength)
            sl_mult = 0.95 + 0.10 * max(0.0, vol_regime - 1.0)
        else:
            base_position = 0.10
            tp_mult = 1.03 + 0.08 * max(0.0, trend_strength)
            sl_mult = 0.97 + 0.08 * max(0.0, vol_regime - 1.0)

        position_size = base_position * quality * liquidity_scalar * funding_scalar * weekend_scalar * session_scalar
        position_size = max(0.04, min(0.18, position_size))

        # Volatility-adjusted exits: wider in calm regimes, tighter in noisy regimes
        atr_scale = max(0.75, min(1.35, 0.012 / max(vol_20, 1e-8)))
        take_profit = 0.004 * tp_mult * atr_scale
        stop_loss = 0.003 * sl_mult * (1.15 if churn > 0.45 else 1.0)

        # Time stop: shorter in choppy / funding-adjacent periods
        max_bars = 4
        if churn > 0.45 or vol_regime > 1.25:
            max_bars = 3
        if near_funding:
            max_bars = 2
        if is_weekend:
            max_bars = min(max_bars, 3)

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
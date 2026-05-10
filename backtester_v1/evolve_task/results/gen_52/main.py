import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Adaptive signal generator for a 15-minute return model.

    Key ideas:
    - Use dynamic thresholds based on spread, volatility, trend, and funding regime.
    - Require multi-factor confirmation from microstructure and price action.
    - Apply asymmetric long/short logic to avoid weak fading conditions.
    - Scale position size and exits by regime quality rather than using fixed values only.
    """

    # --- Safe extraction helpers ---
    get = bar_context.get

    # Core market structure
    close = float(get("close", 1.0) or 1.0)
    spread_bps = float(get("spread_bps", 1.0) or 1.0)
    atr = float(get("atr_14", get("atr", 0.002)) or 0.002)
    vol_5 = float(get("vol_5", 0.001) or 0.001)
    vol_20 = float(get("vol_20", 0.001) or 0.001)
    vol_60 = float(get("vol_60", vol_20) or vol_20)

    # Microstructure
    obi1 = float(get("obi_tau1", 0.0) or 0.0)
    obi3 = float(get("obi_tau3", 0.0) or 0.0)
    obi5 = float(get("obi_tau5", 0.0) or 0.0)
    obi10 = float(get("obi_tau10", 0.0) or 0.0)
    depth_ratio_5 = float(get("depth_ratio_5", 1.0) or 1.0)
    depth_ratio_10 = float(get("depth_ratio_10", 1.0) or 1.0)
    book_pressure_3 = float(get("book_pressure_3", 0.0) or 0.0)
    kyle_lambda = float(get("kyle_lambda_est", 0.0) or 0.0)

    # Price action / momentum
    ret_1 = float(get("ret_1", 0.0) or 0.0)
    ret_3 = float(get("ret_3", 0.0) or 0.0)
    ret_6 = float(get("ret_6", 0.0) or 0.0)
    ret_12 = float(get("ret_12", 0.0) or 0.0)
    ret_48 = float(get("ret_48", 0.0) or 0.0)
    rsi_6 = float(get("rsi_6", get("rsi_14", 50.0)) or 50.0)
    rsi_14 = float(get("rsi_14", 50.0) or 50.0)
    macd_signal = float(get("macd_signal", 0.0) or 0.0)
    bb_pct = float(get("bb_pct", 0.5) or 0.5)
    momentum = float(get("momentum", get("momentum_bar", 0.0)) or 0.0)
    wick_up = float(get("wick_ratio_up", 0.0) or 0.0)
    wick_down = float(get("wick_ratio_down", 0.0) or 0.0)
    volume_ratio = float(get("volume_ratio_5", get("volume_ratio_20", 1.0)) or 1.0)
    vwap_dev = float(get("vwap_dev", 0.0) or 0.0)
    autocorr_5 = float(get("autocorr_5", 0.0) or 0.0)
    skew = float(get("skew_20", 0.0) or 0.0)
    kurt = float(get("kurt_20", 0.0) or 0.0)
    trend_strength = float(get("trend_strength", 0.0) or 0.0)

    # Macro / timing
    funding_rate = float(get("funding_rate", 0.0) or 0.0)
    funding_ma = float(get("funding_8h_ma", funding_rate) or funding_rate)
    minutes_to_funding = float(get("minutes_to_funding", 9999.0) or 9999.0)

    # Session regime
    is_us = bool(get("is_us_session", False))
    is_asia = bool(get("is_asia_session", False))
    is_weekend = bool(get("is_weekend", False))

    # --- Derived regime features ---
    atr_pct = atr / close if close > 0 else atr
    vol_regime = max(vol_5, vol_20, vol_60)
    vol_slope = vol_5 - vol_20
    vol_expansion = 1.0 if vol_slope > 0 else 0.0

    obi_combo = 0.35 * obi1 + 0.65 * obi5
    obi_stack = (0.25 * obi1) + (0.25 * obi3) + (0.30 * obi5) + (0.20 * obi10)

    funding_pressure = funding_rate - funding_ma
    impact_penalty = max(0.0, min(1.0, abs(kyle_lambda) * 10.0))
    spread_penalty = max(0.0, min(1.0, spread_bps / 12.0))

    # --- Dynamic thresholding ---
    base_threshold = 0.00185

    # Wider spreads / higher impact require stronger model edge
    base_threshold *= (1.0 + 0.18 * spread_penalty + 0.12 * impact_penalty)

    # Higher volatility requires a slightly larger edge to avoid noise
    if atr_pct > 0:
        base_threshold *= (1.0 + max(0.0, min(0.25, (atr_pct - 0.0045) * 14.0)))

    # Session adjustments: US slightly more permissive, Asia more selective
    if is_us:
        base_threshold *= 0.95
    elif is_asia:
        base_threshold *= 1.06

    # Weekend / pre-funding caution
    if is_weekend:
        base_threshold *= 1.08
    if minutes_to_funding <= 45.0:
        base_threshold *= 1.05

    # Trend alignment can reduce threshold modestly
    if predicted_return > 0 and trend_strength > 0.35:
        base_threshold *= 0.92
    elif predicted_return < 0 and trend_strength < -0.35:
        base_threshold *= 0.92

    # --- Signal logic with asymmetric filters ---
    signal = 0

    # Long setup
    long_setup = (
        predicted_return > base_threshold and
        obi_combo > -0.12 and
        obi_stack > -0.18 and
        depth_ratio_5 > 0.92 and
        depth_ratio_10 > 0.90 and
        spread_bps < 10.0 and
        vwap_dev < 0.018 and
        rsi_6 < 74.0 and
        rsi_14 < 71.0 and
        macd_signal > -0.0005 and
        bb_pct < 0.92 and
        momentum > -0.15 and
        book_pressure_3 > -0.25 and
        (funding_pressure < 0.0006 or predicted_return > base_threshold * 1.35)
    )

    # Short setup
    short_setup = (
        predicted_return < -base_threshold and
        obi_combo < 0.12 and
        obi_stack < 0.18 and
        depth_ratio_5 > 0.92 and
        depth_ratio_10 > 0.90 and
        spread_bps < 10.0 and
        vwap_dev > -0.018 and
        rsi_6 > 26.0 and
        rsi_14 > 29.0 and
        macd_signal < 0.0005 and
        bb_pct > 0.08 and
        momentum < 0.15 and
        book_pressure_3 < 0.25 and
        (funding_pressure > -0.0006 or predicted_return < -base_threshold * 1.35)
    )

    # Mean-reversion protection: avoid fading strong trend with adverse autocorrelation
    if long_setup:
        if not (trend_strength < -0.55 and autocorr_5 < -0.15 and ret_12 < 0 and ret_48 < 0):
            signal = 1
    elif short_setup:
        if not (trend_strength > 0.55 and autocorr_5 < -0.15 and ret_12 > 0 and ret_48 > 0):
            signal = -1

    # Extra vetoes for poor execution conditions
    if signal != 0:
        if spread_bps >= 12.0 or impact_penalty > 0.95:
            signal = 0
        elif is_weekend and abs(predicted_return) < base_threshold * 1.15:
            signal = 0

    # --- Position sizing ---
    if signal != 0:
        confidence = abs(predicted_return) / max(base_threshold, 1e-6)

        # Convert confidence to a bounded multiplier
        conf_mult = 0.85 + 0.22 * min(2.0, confidence)
        conf_mult = max(0.85, min(1.35, conf_mult))

        # Quality score from regime alignment
        quality = 1.0
        quality *= 1.0 + 0.10 * max(-1.0, min(1.0, abs(obi_stack)))
        quality *= 1.0 + 0.05 * max(-1.0, min(1.0, abs(trend_strength)))
        quality *= 1.0 - 0.10 * spread_penalty
        quality *= 1.0 - 0.08 * impact_penalty

        # Reduce size in unstable volatility expansions
        if vol_expansion and atr_pct > 0.0065:
            quality *= 0.90

        # Session-specific scaling
        if is_us:
            quality *= 1.05
        elif is_asia:
            quality *= 0.95

        position_size = 0.105 * conf_mult * quality
        position_size = max(0.07, min(0.16, position_size))
    else:
        position_size = 0.0

    # --- Risk management ---
    if signal != 0:
        # Volatility-adjusted exits with asymmetry
        vol_scale = max(0.85, min(1.20, 1.0 + (atr_pct - 0.0045) * 18.0))
        spread_scale = max(0.90, min(1.10, 1.0 - 0.01 * spread_bps))
        trend_scale = max(0.92, min(1.08, 1.0 + 0.04 * abs(trend_strength)))

        take_profit = 0.0040 * vol_scale * trend_scale
        stop_loss = 0.0030 * vol_scale / max(0.90, min(1.08, spread_scale))

        # Asymmetric tweaking: let winners run a bit more in strong trend, tighten in chop
        if abs(trend_strength) > 0.60 and autocorr_5 > 0.05:
            take_profit *= 1.10
            stop_loss *= 0.95
        elif abs(trend_strength) < 0.20 and autocorr_5 < 0.0:
            take_profit *= 0.95
            stop_loss *= 1.05

        # Funding proximity risk: reduce target modestly before funding events
        if minutes_to_funding <= 30.0:
            take_profit *= 0.96
            stop_loss *= 0.98

        max_bars = 4
        if is_us and trend_strength > 0.40:
            max_bars = 5
        elif abs(trend_strength) > 0.70 and autocorr_5 > 0.10:
            max_bars = 6
        elif is_asia and abs(trend_strength) < 0.20:
            max_bars = 3
    else:
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

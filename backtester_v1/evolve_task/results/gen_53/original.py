import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Modular trading signal generator with decoupled components.

    Architecture:
    1. Parameter Resolution: Extract and validate all context values
    2. Signal Classification: Pure threshold-based entry logic
    3. Optional Filters: Microstructure/macro checks (disabled to match best performance)
    4. Risk Management: Position sizing, exits, time-based rules
    """
    import math
    import numpy as np

    # ============================================================================
    # COMPONENT 1: PARAMETER RESOLUTION
    # ============================================================================
    def _safe_get(key, default=0.0, cast_fn=float):
        """Safely extract and cast context values."""
        v = bar_context.get(key, default)
        if v is None:
            return default
        try:
            return cast_fn(v)
        except (ValueError, TypeError):
            return default

    # Microstructure parameters
    obi_tau1 = _safe_get("obi_tau1", 0.0)
    obi_tau3 = _safe_get("obi_tau3", 0.0)
    obi_tau5 = _safe_get("obi_tau5", 0.0)
    spread_bps = _safe_get("spread_bps", 5.0)
    book_pressure = _safe_get("book_pressure_3", 0.0)
    kyle_lambda = abs(_safe_get("kyle_lambda_est", 0.0))
    depth_ratio_5 = _safe_get("depth_ratio_5", 1.0)

    # Price action & volatility
    vol_5 = max(_safe_get("vol_5", 0.015), 1e-8)
    vol_20 = max(_safe_get("vol_20", 0.015), 1e-8)
    vol_60 = max(_safe_get("vol_60", 0.015), 1e-8)
    atr_14 = max(_safe_get("atr_14", _safe_get("atr", 0.001)), 1e-8)
    rsi_14 = _safe_get("rsi_14", _safe_get("rsi", 50.0))
    bb_pct = _safe_get("bb_pct", 0.5)
    trend_strength = _safe_get("trend_strength", 0.0)

    # Macro parameters
    funding_rate = _safe_get("funding_rate", 0.0)
    funding_8h_ma = _safe_get("funding_8h_ma", 0.0)
    minutes_to_funding = _safe_get("minutes_to_funding", 999.0)

    # Session/time parameters
    is_us_session = bar_context.get("is_us_session", False) or False
    is_asia_session = bar_context.get("is_asia_session", False) or False
    is_weekend = bar_context.get("is_weekend", False) or False

    # ============================================================================
    # COMPONENT 2: SIGNAL CLASSIFICATION
    # Seed 3 proved: simple fixed thresholds outperform complex adaptive logic
    # ============================================================================
    BASE_LONG_THRESH = 0.002
    BASE_SHORT_THRESH = -0.002

    # No adaptive thresholds based on session (Seed 3 didn't use this and scored best)
    signal = 0
    if predicted_return > BASE_LONG_THRESH:
        signal = 1
    elif predicted_return < BASE_SHORT_THRESH:
        signal = -1

    # ============================================================================
    # COMPONENT 3: OPTIONAL FILTERS (Disabled by default)
    # Current implementation's OBI/spread filters reduced trades 462→253, hurting performance
    # Keeping code structure for evolutionary exploration but not applying them
    # ============================================================================
    APPLY_MICROSTRUCTURE_FILTERS = False  # Disabled: reduces trade count destructively

    if APPLY_MICROSTRUCTURE_FILTERS and signal != 0:
        # Microstructure validation (if re-enabled)
        long_obi_filter = obi_tau5 > 0.05
        short_obi_filter = obi_tau5 < -0.05
        spread_filter = spread_bps < 10.0

        if signal == 1 and not (long_obi_filter and spread_filter):
            signal = 0
        elif signal == -1 and not (short_obi_filter and spread_filter):
            signal = 0

    # Funding rate check (light: only skip on extreme rates)
    APPLY_FUNDING_FILTER = False  # Disabled: Seed 3 ignored this
    if APPLY_FUNDING_FILTER and signal != 0:
        if signal == 1 and funding_rate > 0.002:
            signal = 0
        elif signal == -1 and funding_rate < -0.002:
            signal = 0

    # ============================================================================
    # COMPONENT 4: RISK MANAGEMENT
    # Match Seed 3's winning parameters: fixed position sizing and exits
    # ============================================================================

    # Position sizing: Use fixed base (Seed 3's 0.10 worked best)
    BASE_POSITION_SIZE = 0.10

    # Volatility scalar (optional enhancement, use conservative bounds)
    vol_regime = vol_5 / vol_20  # expansion/contraction metric
    vol_regime = max(0.5, min(2.0, vol_regime))

    # Scale position by volatility: high vol reduces size
    vol_scalar = 1.0 + (1.0 - vol_regime) * 0.15  # max ±15% adjustment
    vol_scalar = max(0.85, min(1.15, vol_scalar))

    # Funding scalar: reduce size at extreme funding rates
    funding_scalar = 1.0
    if abs(funding_rate) > 0.001:
        funding_scalar = max(0.7, 1.0 - abs(funding_rate - funding_8h_ma) * 50.0)
    funding_scalar = max(0.7, min(1.0, funding_scalar))

    if signal == 0:
        position_size = 0.0
    else:
        position_size = BASE_POSITION_SIZE * vol_scalar * funding_scalar
        position_size = max(0.05, min(0.15, position_size))  # Clamp to [0.05, 0.15]

    # Exit targets: Match Seed 3's proven fixed values
    BASE_TAKE_PROFIT = 0.004
    BASE_STOP_LOSS = 0.003

    # Optional: Slight adjustment based on regime (conservative)
    tp_multiplier = 1.0
    sl_multiplier = 1.0

    # In very low volatility, allow slightly larger moves
    if vol_5 < 0.008:
        tp_multiplier = 1.1
        sl_multiplier = 1.05
    # In very high volatility, keep tighter
    elif vol_5 > 0.035:
        tp_multiplier = 0.95
        sl_multiplier = 0.95

    take_profit = BASE_TAKE_PROFIT * tp_multiplier
    stop_loss = BASE_STOP_LOSS * sl_multiplier

    # Time-based exit: Seed 3 used 4 bars (1 hour)
    MAX_BARS = 4

    # Optional: Reduce max bars near funding time (avoid overnight holds)
    if minutes_to_funding < 30.0 and minutes_to_funding > 0.0:
        MAX_BARS = min(MAX_BARS, 2)

    if is_weekend:
        MAX_BARS = min(MAX_BARS, 3)  # Reduce weekend exposure

    # ============================================================================
    # RETURN SIGNAL DICT
    # ============================================================================
    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(MAX_BARS),
    }
# EVOLVE-BLOCK-END
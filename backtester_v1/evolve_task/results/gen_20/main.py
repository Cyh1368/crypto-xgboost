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

    # 1. Extract Core Context
    # Microstructure
    obi1 = float(_get("obi_tau1"))
    obi3 = float(_get("obi_tau3"))
    obi5 = float(_get("obi_tau5"))
    book_pressure = float(_get("book_pressure_3"))
    spread_bps = float(_get("spread_bps"))

    # Price Action & Volatility
    vol5 = max(float(_get("vol_5")), 1e-8)
    vol20 = max(float(_get("vol_20")), 1e-8)
    vol60 = max(float(_get("vol_60")), 1e-8)
    atr = float(_get("atr_14", _get("atr", 0.001)))
    rsi14 = float(_get("rsi_14", 50.0))
    rsi6 = float(_get("rsi_6", 50.0))
    trend = float(_get("trend_strength"))

    # Macro & Time
    funding = float(_get("funding_rate"))
    funding_ma = float(_get("funding_8h_ma"))
    is_us = bool(_get("is_us_session", False))
    is_weekend = bool(_get("is_weekend", False))

    # 2. Regime & Quality Metrics
    # micro_quality: positive means book supports long, negative supports short
    micro_quality = 0.3 * obi1 + 0.25 * obi3 + 0.25 * obi5 + 0.2 * book_pressure

    # vol_regime: > 1.0 means volatility is expanding
    vol_regime = max(0.5, min(2.0, vol5 / vol20))

    # vol_expansion: measure if volatility is spiking (vol5 > vol60)
    vol_expansion = vol5 / max(vol60, 1e-8)

    # signal_conviction: how strong is the predicted return relative to recent volatility
    signal_conviction = abs(predicted_return) / max(vol5, 0.001)

    # 3. Signal Generation
    # Threshold scales with volatility regime to capture more trades in low-vol and filter noise in high-vol
    LONG_THRESH = 0.0018 * (0.9 + 0.2 * vol_regime)
    SHORT_THRESH = -0.0018 * (0.9 + 0.2 * vol_regime)

    signal = 0
    # Lenient filters to increase trade frequency while avoiding extreme toxic flow or high costs
    # Add RSI filter: avoid extreme overbought (>75) and oversold (<25) on fast RSI
    rsi_long_ok = rsi6 < 75.0
    rsi_short_ok = rsi6 > 25.0

    if predicted_return > LONG_THRESH:
        if micro_quality > -0.8 and funding < 0.002 and spread_bps < 10.0 and rsi_long_ok:
            signal = 1
    elif predicted_return < SHORT_THRESH:
        if micro_quality < 0.8 and funding > -0.002 and spread_bps < 10.0 and rsi_short_ok:
            signal = -1

    # 4. Position Sizing
    # Dynamic sizing based on signal conviction and microstructure quality
    if signal == 0:
        position_size = 0.0
    else:
        # Base position size
        base_size = 0.11

        # Conviction multiplier: higher conviction signals get larger positions
        conviction_mult = 1.0 + min(0.3, signal_conviction * 2.0)

        # Micro quality multiplier: better microstructure = larger position
        quality_mult = 1.0 + (micro_quality / 2.0) * (0.2 if signal == 1 else -0.2)
        quality_mult = max(0.7, min(1.3, quality_mult))

        # Funding adjustment: reduce size if funding is extreme
        funding_mult = 1.0
        if abs(funding) > 0.0008:
            funding_mult = max(0.6, 1.0 - abs(funding) * 500.0)

        position_size = base_size * conviction_mult * quality_mult * funding_mult
        position_size = max(0.06, min(0.16, position_size))  # clamp between 6% and 16%

    # 5. Dynamic Exits
    # Tighter stops in high volatility or poor microstructure
    base_stop = 0.003
    vol_stop_mult = 1.0 + (vol_expansion - 1.0) * 0.3  # expand stop if vol spiking
    micro_stop_mult = 1.0 - (abs(micro_quality) / 2.0) * 0.2  # tighter stop if poor quality

    stop_loss = base_stop * vol_stop_mult * micro_stop_mult
    stop_loss = max(0.0015, min(0.005, stop_loss))  # clamp between 0.15% and 0.5%

    take_profit = 0.004

    # 6. Time-based Exit
    max_bars = 4

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
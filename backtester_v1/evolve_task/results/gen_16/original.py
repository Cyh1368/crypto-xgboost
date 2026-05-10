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
    book_pressure = float(_get("book_pressure_3"))
    spread_bps = float(_get("spread_bps"))

    # Price Action & Volatility
    vol5 = max(float(_get("vol_5")), 1e-8)
    vol20 = max(float(_get("vol_20")), 1e-8)
    atr = float(_get("atr_14", _get("atr", 0.001)))
    rsi14 = float(_get("rsi_14", 50.0))
    trend = float(_get("trend_strength"))

    # Macro & Time
    funding = float(_get("funding_rate"))
    funding_ma = float(_get("funding_8h_ma"))
    is_us = bool(_get("is_us_session", False))
    is_weekend = bool(_get("is_weekend", False))

    # 2. Regime & Quality Metrics
    # micro_quality: positive means book supports long, negative supports short
    micro_quality = 0.4 * obi1 + 0.3 * obi3 + 0.3 * book_pressure

    # vol_regime: > 1.0 means volatility is expanding
    vol_regime = max(0.5, min(2.0, vol5 / vol20))

    # 3. Signal Generation (Thresholds slightly more aggressive than seed)
    LONG_THRESH = 0.0019
    SHORT_THRESH = -0.0019

    signal = 0
    # Relaxed filters to increase trade frequency while avoiding toxic flow
    if predicted_return > LONG_THRESH:
        if micro_quality > -0.7 and funding < 0.001:
            signal = 1
    elif predicted_return < SHORT_THRESH:
        if micro_quality < 0.7 and funding > -0.001:
            signal = -1

    # 4. Position Sizing
    # Standardized size slightly above seed to maximize total return
    if signal == 0:
        position_size = 0.0
    else:
        position_size = 0.11

    # 5. Fixed Exits from high-performing seed
    take_profit = 0.004
    stop_loss = 0.003

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
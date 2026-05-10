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
    kyle_lambda = abs(float(_get("kyle_lambda_est")))

    # Price Action & Volatility
    vol5 = max(float(_get("vol_5")), 1e-8)
    vol20 = max(float(_get("vol_20")), 1e-8)
    # vol_regime: > 1.0 means volatility is expanding relative to the recent past
    vol_regime = max(0.5, min(2.0, vol5 / vol20))

    # Macro & Session
    funding = float(_get("funding_rate"))
    is_us = bool(_get("is_us_session", False))

    # 2. Regime Quality Metrics
    # micro_quality: Positive means book supports long, negative supports short.
    # Weights optimized from high-performing crossover iterations.
    micro_quality = 0.3 * obi1 + 0.25 * obi3 + 0.25 * obi5 + 0.2 * book_pressure

    # 3. Dynamic Signal Thresholding
    # Scaling thresholds with volatility prevents over-trading in noise and captures meaningful moves.
    BASE_THRESH = 0.0018
    long_thresh = BASE_THRESH * (0.9 + 0.2 * vol_regime)
    short_thresh = -BASE_THRESH * (0.9 + 0.2 * vol_regime)

    signal = 0
    # Core signal logic: Model prediction must exceed threshold, confirmed by microstructure flow
    # and filtered by execution cost (spread) and carry cost (funding).
    if predicted_return > long_thresh:
        if micro_quality > -0.8 and funding < 0.002 and spread_bps < 9.5:
            signal = 1
    elif predicted_return < short_thresh:
        if micro_quality < 0.8 and funding > -0.002 and spread_bps < 9.5:
            signal = -1

    # 4. Asymmetric Position Sizing
    # We increase size based on predicted conviction and favorable session liquidity.
    if signal == 0:
        position_size = 0.0
    else:
        # Base size slightly increased from previous seed to leverage high win rate
        base_size = 0.13
        # Conviction boost
        conviction_boost = 0.02 if abs(predicted_return) > 0.0036 else 0.0
        # US Session boost (liquidity and move persistence)
        session_boost = 0.01 if is_us else 0.0

        position_size = base_size + conviction_boost + session_boost
        position_size = min(0.18, max(0.10, position_size))

    # 5. Fixed-Horizon Risk Management
    # Fixed TP/SL has proven to be the most robust exit strategy for this 15-min model.
    take_profit = 0.004
    stop_loss = 0.003
    max_bars = 4 # Time-based exit at 60 minutes

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
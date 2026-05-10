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

    # Price Action & Volatility
    vol5 = max(float(_get("vol_5")), 1e-8)
    vol20 = max(float(_get("vol_20")), 1e-8)
    vol60 = max(float(_get("vol_60")), 1e-8)
    # vol_regime: > 1.0 means volatility is expanding relative to the recent past
    vol_regime = max(0.5, min(2.0, vol5 / vol20))
    # vol_trend: long-term volatility regime for exit timing
    vol_trend = max(0.5, min(2.0, vol5 / vol60))

    # Macro & Session
    funding = float(_get("funding_rate"))
    is_us = bool(_get("is_us_session", False))

    # 2. Regime Quality Metrics
    # micro_quality: Positive means book supports long, negative supports short.
    # Weights optimized from high-performing crossover iterations.
    micro_quality = 0.3 * obi1 + 0.25 * obi3 + 0.25 * obi5 + 0.2 * book_pressure

    # 3. Dynamic Signal Thresholding
    # Elastic thresholding: base scaled by volatility impulse improves signal quality
    # while increasing trade frequency in meaningful vol regimes
    BASE_THRESH = 0.00172
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_regime))
    current_threshold = BASE_THRESH * elastic_multiplier

    signal = 0
    signal_conviction = 0.0
    # Core signal logic: Model prediction must exceed elastic threshold, confirmed by microstructure
    # and filtered by execution cost (spread) and carry cost (funding).
    if predicted_return > current_threshold:
        if micro_quality > -0.85 and funding < 0.0025 and spread_bps < 10.5:
            signal = 1
            signal_conviction = predicted_return / current_threshold
    elif predicted_return < -current_threshold:
        if micro_quality < 0.85 and funding > -0.0025 and spread_bps < 10.5:
            signal = -1
            signal_conviction = abs(predicted_return) / current_threshold

    # 4. Asymmetric Position Sizing
    # Scale sizing by conviction strength and session liquidity for better capital efficiency
    if signal == 0:
        position_size = 0.0
    else:
        # Base size from proven high-performing seed
        base_size = 0.145
        # Conviction bonus: scale with signal strength relative to threshold
        conviction_bonus = min(0.035, max(0, (signal_conviction - 1.0) * 0.06))
        # US Session boost (liquidity and move persistence)
        session_bonus = 0.01 if is_us else 0.0
        # Funding adjustment: mild penalty for unfavorable carry
        funding_filter = 1.0
        if (signal == 1 and funding > 0.0015) or (signal == -1 and funding < -0.0015):
            funding_filter = 0.90

        position_size = (base_size + conviction_bonus + session_bonus) * funding_filter
        position_size = max(0.10, min(0.18, position_size))

    # 5. Volatility-Elastic Risk Management
    # Scale exits by volatility regime: expand in high-vol, contract in low-vol
    tp_elasticity = 0.94 + (0.06 * min(2.0, vol_trend))
    sl_elasticity = 0.97 + (0.03 * min(2.0, vol_regime))

    take_profit = 0.0040 * tp_elasticity
    stop_loss = 0.0030 * sl_elasticity

    # Dynamic time-based exit: exit faster if volatility trending down (stagnation risk)
    max_bars = 4
    if vol_trend < 0.8:
        max_bars = 3

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
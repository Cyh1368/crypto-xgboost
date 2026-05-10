import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    High-frequency signal generator balancing model predictions with microstructure
    safety filters and session-specific thresholds.
    """
    # 1. Context Extraction
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    obi_tau5 = bar_context.get('obi_tau5', 0.0)
    spread_bps = bar_context.get('spread_bps', 1.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    atr = bar_context.get('atr_14', 0.002)
    close = bar_context.get('close', 1.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    rv_ratio = bar_context.get('realized_vol_ratio', 1.0)

    # 2. Dynamic Entry Thresholds
    # Higher thresholds to improve win rate and Sharpe ratio.
    base_thresh = 0.00195
    if is_us:
        base_thresh = 0.00185 # Capture high-volume moves
    elif is_asia:
        base_thresh = 0.0022  # More selective during lower volatility

    signal = 0
    # 3. Filtering Strategy (Stricter Microstructure & Liquidity)
    # rv_ratio > 1.8 often indicates unstable/reversing regimes.
    vol_regime_ok = rv_ratio < 1.7

    if predicted_return > base_thresh and vol_regime_ok:
        # Long filter: Tighter OBI and VWAP constraints
        if obi_tau5 > -0.25 and vwap_dev < 0.009 and spread_bps < 5.0:
            if funding < 0.0004:
                signal = 1

    elif predicted_return < -base_thresh and vol_regime_ok:
        # Short filter: Avoid entering if OBI is biased for lungs/spread is wide
        if obi_tau5 < 0.25 and vwap_dev > -0.009 and spread_bps < 5.0:
            if funding > -0.0004:
                signal = -1

    # 4. Position Sizing
    # Weighted by model return relative to 0.2% benchmark.
    if signal != 0:
        confidence = abs(predicted_return) / 0.002
        position_size = 0.11 * confidence
        # Moderate sizing with caps to control max drawdown
        position_size = max(0.08, min(0.14, position_size))

        # Reduce size in high-volatility environments relative to price as per ATR
        atr_pct = (atr / close) if close > 0 else 0.002
        if atr_pct > 0.007:
            position_size *= 0.85
    else:
        position_size = 0.0

    # 5. Risk Management (Optimized TP/SL)
    # Returning to robust 4:3 reward-to-risk ratio from successful seed.
    take_profit = 0.0040
    stop_loss = 0.0030

    # 6. Adaptive Hold Time
    if abs(trend) > 0.65 and is_us:
        max_bars = 5
    else:
        max_bars = 4

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
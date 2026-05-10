import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Regime-weighted liquidity strategy focusing on order-book persistence,
    persistence-based time exits, and asymmetric risk filters.
    """
    # 1. Parameter Extraction
    close = bar_context.get('close', 1.0)
    atr = bar_context.get('atr_14', 0.001)

    # Microstructure
    obi_1 = bar_context.get('obi_tau1', 0.0)
    obi_5 = bar_context.get('obi_tau5', 0.0)
    obi_10 = bar_context.get('obi_tau10', 0.0)
    avg_obi = (obi_1 + obi_5 + obi_10) / 3.0

    spread = bar_context.get('spread_bps', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)

    # Contextual Regimes
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    funding = bar_context.get('funding_rate', 0.0)

    # Sessions
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)

    # 2. Threshold and Signal Generation
    # Session-adaptive thresholds with a liquidity penalty to improve win rate.
    base_thresh = 0.00175
    if is_us:
        base_thresh = 0.00168
    elif is_asia:
        base_thresh = 0.00195

    # Liquidity penalty: increase threshold if the book is thin (Kyle's Lambda)
    effective_thresh = base_thresh + min(0.00025, kyle_lambda * 35.0)
    signal = 0

    if predicted_return > effective_thresh:
        # Long Filter: Tightened OBI and VWAP constraints to ensure quality
        if avg_obi > -0.55 and funding < 0.0012 and spread < 10.0:
            if vwap_dev < 0.022:
                signal = 1

    elif predicted_return < -effective_thresh:
        # Short Filter: Symmetric tightened constraints
        if avg_obi < 0.55 and funding > -0.0012 and spread < 10.0:
            if vwap_dev > -0.022:
                signal = -1

    # 3. Position Sizing
    # Dynamic sizing based on confidence and volatility-adjusted scaling.
    if signal != 0:
        base_size = 0.142
        conf_boost = min(1.2, abs(predicted_return) / effective_thresh)
        position_size = base_size * conf_boost

        # Volatility scaling: reduce size in extreme volatility
        atr_pct = (atr / close) if close > 0 else 0.0015
        if atr_pct > 0.0085:
            position_size *= 0.85

        position_size = max(0.09, min(0.19, position_size))

        # 4. Target Setting (TP / SL)
        # Asymmetric TP/SL with trend-following adjustments.
        tp_base = 0.0042
        sl_base = 0.0031

        if (signal == 1 and trend > 0.4) or (signal == -1 and trend < -0.4):
            tp_base += 0.0006
            sl_base -= 0.0001

        take_profit = tp_base
        stop_loss = sl_base

        # 5. Time-Based Exit Strategy (Max Bars)
        # Optimized for 15m model persistence; longer holds in trending or US sessions.
        if (is_us and abs(trend) > 0.35) or abs(autocorr) > 0.18:
            max_bars = 5
        else:
            max_bars = 4

    else:
        position_size = 0.0
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
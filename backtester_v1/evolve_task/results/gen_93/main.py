import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Combines session-specific thresholding with relaxed microstructure filters
    to maximize trade frequency and capture alpha in high-win-rate regimes.
    """
    # 1. Context Extraction
    close = bar_context.get('close', 1.0)
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    obi = bar_context.get('obi_tau5', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    atr = bar_context.get('atr_14', 0.002)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)

    # 2. Dynamic Entry Thresholds
    # Lowered thresholds slightly to capture more alpha while session-adjusting.
    base_thresh = 0.00174
    if is_us:
        base_thresh = 0.00168  # Aggressive US entry
    elif is_asia:
        base_thresh = 0.00195  # Conservative Asia entry

    # Combined liquidity and book pressure filter
    pressure = bar_context.get('book_pressure_3', 0.0)
    micro_score = (obi + (pressure / 100.0)) / 2.0

    # Add a small penalty for poor liquidity (Kyle's Lambda)
    liquidity_penalty = min(0.00025, kyle_lambda * 30.0)
    effective_thresh = base_thresh + liquidity_penalty

    # 3. Execution Signal Logic
    signal = 0
    if predicted_return > effective_thresh:
        # Long filter: Micro-alignment and funding safety
        if micro_score > -0.65 and vwap_dev < 0.026 and spread < 11.0:
            if funding < 0.0018:
                signal = 1

    elif predicted_return < -effective_thresh:
        # Short filter: Micro-alignment and funding safety
        if micro_score < 0.65 and vwap_dev > -0.026 and spread < 11.0:
            if funding > -0.0018:
                signal = -1

    # 4. Position Sizing
    # Increased base size to 0.15 with a wider multiplier for high-confidence trades.
    if signal != 0:
        confidence = abs(predicted_return) / effective_thresh
        position_size = 0.15 * min(1.25, confidence)
        # Final safety bounds
        position_size = max(0.10, min(0.20, position_size))

        # Volatility check: reduce size if ATR is extremely high
        atr_pct = (atr / close) if close > 0 else 0.002
        if atr_pct > 0.012:
            position_size *= 0.80
    else:
        position_size = 0.0

    # 5. Risk Management (Dynamic TP/SL)
    if signal != 0:
        # Base R/R Targets
        take_profit = 0.0042
        stop_loss = 0.0032

        # Funding Bonus: If we earn the funding rate, extend the target
        if (signal == 1 and funding < 0) or (signal == -1 and funding > 0):
            take_profit += 0.0005

        # Trend and Autocorrelation Bias
        if abs(trend) > 0.45 or abs(autocorr) > 0.2:
            take_profit += 0.0004
            stop_loss -= 0.0001

        # 6. Adaptive Hold Time
        # Session and trend based persistence
        if is_us or abs(trend) > 0.35:
            max_bars = 5
        elif is_asia and abs(autocorr) < 0.05:
            max_bars = 3 # Fast exit in choppy Asia sessions
        else:
            max_bars = 4
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
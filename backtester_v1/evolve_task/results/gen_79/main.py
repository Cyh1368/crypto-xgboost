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
    # Base threshold from the 90.84-score program, adjusted by session.
    base_thresh = 0.0018
    if is_us:
        base_thresh = 0.00172  # Lower threshold for high-volatility US session
    elif is_asia:
        base_thresh = 0.00205  # Higher threshold for lower-volatility Asia session
    
    # Add a small penalty for poor liquidity (Kyle's Lambda)
    liquidity_penalty = min(0.0002, kyle_lambda * 25.0)
    effective_thresh = base_thresh + liquidity_penalty

    # 3. Execution Signal Logic (Relaxed Filters)
    signal = 0
    if predicted_return > effective_thresh:
        # Long filter: Relaxed boundaries to recapture trade volume
        if obi > -0.7 and vwap_dev < 0.025 and spread < 10.0:
            if funding < 0.0015:
                signal = 1
                
    elif predicted_return < -effective_thresh:
        # Short filter: Relaxed boundaries to recapture trade volume
        if obi < 0.7 and vwap_dev > -0.025 and spread < 10.0:
            if funding > -0.0015:
                signal = -1

    # 4. Position Sizing
    # Confidence-weighted sizing anchored at 0.135
    if signal != 0:
        confidence = abs(predicted_return) / effective_thresh
        position_size = 0.135 * min(1.2, confidence)
        # Final safety bounds
        position_size = max(0.10, min(0.185, position_size))
        
        # Volatility check: reduce size if ATR is extremely high
        atr_pct = (atr / close) if close > 0 else 0.002
        if atr_pct > 0.01:
            position_size *= 0.85
    else:
        position_size = 0.0

    # 5. Risk Management (Optimized TP/SL)
    # Anchored to the robust 0.004/0.003 baseline with a slight expansion.
    if signal != 0:
        take_profit = 0.0041
        stop_loss = 0.0031
        
        # 6. Adaptive Hold Time
        # Default 4 bars, extending to 5 for trending or high-persistence states.
        if (is_us and abs(trend) > 0.35) or abs(autocorr) > 0.18:
            max_bars = 5
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
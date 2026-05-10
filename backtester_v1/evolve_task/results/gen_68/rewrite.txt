import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Regime-Adaptive Microflow Logic:
    Integrates market microstructure depth with price-action regime filters.
    """
    import math

    def _get_val(key, default=0.0):
        val = bar_context.get(key, default)
        return float(val) if val is not None else default

    # 1. Market State Evaluation
    # Volatility ratios to determine if we are in an expansion or contraction phase
    v5 = max(_get_val("vol_5", 0.012), 1e-9)
    v20 = max(_get_val("vol_20", 0.012), 1e-9)
    v60 = max(_get_val("vol_60", 0.012), 1e-9)
    
    vol_impulse = v5 / v20
    vol_trend = v5 / v60
    
    # Trend persistence indicators
    trend_str = _get_val("trend_strength", 0.0)
    auto_corr = _get_val("autocorr_5", 0.0)
    rsi = _get_val("rsi_14", 50.0)
    
    # 2. Advanced Microstructure Filter
    # Combine OBI across multiple horizons and incorporate book pressure
    obi_composite = (
        0.25 * _get_val("obi_tau1") +
        0.20 * _get_val("obi_tau3") +
        0.15 * _get_val("obi_tau5") +
        0.10 * _get_val("obi_tau10") +
        0.30 * _get_val("book_pressure_3")
    )
    
    # Use Kyle's Lambda to estimate market impact cost
    kyle_lambda = abs(_get_val("kyle_lambda_est", 0.0))
    impact_filter = 1.0 if kyle_lambda < 0.05 else 0.8 # Scale down if high impact
    
    # 3. Dynamic Thresholding
    # Adjust base threshold (0.00168) by vol_impulse to ensure signal quality in high vol
    base_threshold = 0.00168
    dynamic_threshold = base_threshold * (0.86 + 0.18 * min(2.5, vol_impulse))
    
    signal = 0
    if predicted_return > dynamic_threshold:
        # Long entry: price prediction + book flow + funding cost filter
        if obi_composite > -0.75 and _get_val("funding_rate") < 0.0022:
            if _get_val("spread_bps") < 11.0:
                signal = 1
    elif predicted_return < -dynamic_threshold:
        # Short entry: price prediction + book flow + funding cost filter
        if obi_composite < 0.75 and _get_val("funding_rate") > -0.0022:
            if _get_val("spread_bps") < 11.0:
                signal = -1

    # 4. Position Sizing
    if signal == 0:
        position_size = 0.0
    else:
        # Base size scaled by conviction and market state
        base_size = 0.152
        conviction = abs(predicted_return) / dynamic_threshold
        
        # Momentum Bonus: Increase size if rsi indicates room to run
        rsi_bonus = 0.0
        if signal == 1 and rsi < 58:
            rsi_bonus = 0.015
        elif signal == -1 and rsi > 42:
            rsi_bonus = 0.015
            
        # Session/Macro adjustment
        session_bonus = 0.008 if bar_context.get("is_us_session") else -0.005
        if bar_context.get("is_weekend"): session_bonus -= 0.01
        
        position_size = (base_size + rsi_bonus + session_bonus) * min(1.1, conviction)
        position_size = max(0.11, min(0.19, position_size)) * impact_filter

    # 5. Volatility-Elastic Exit Strategy
    # Using asymmetric risk:reward and dynamic hold times
    tp_base = 0.0041
    sl_base = 0.0029
    
    # Scale TP/SL based on market volatility trend
    tp = tp_base * (0.95 + 0.08 * min(2.0, vol_trend))
    sl = sl_base * (0.97 + 0.05 * min(2.0, vol_impulse))
    
    # Final Max Bars logic based on trend persistence (autocorr_5)
    max_bars = 4
    if auto_corr > 0.35 or abs(trend_str) > 0.4:
        max_bars = 5 # Ride the trend longer
    elif vol_impulse < 0.75:
        max_bars = 3 # Exit faster in low-activity stagnating markets

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(round(tp, 6)),
        "stop_loss": float(round(sl, 6)),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
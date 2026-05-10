import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    A regime-adaptive signal generator that modulates entry thresholds and 
    risk parameters based on price-action persistence and microstructure quality.
    """

    def _safe_get(key, default=0.0):
        v = bar_context.get(key, default)
        return default if v is None else float(v)

    # 1. PARAMETER EXTRACTION
    # Volatility and Price Action
    v5 = max(_safe_get("vol_5", 0.015), 1e-8)
    v20 = max(_safe_get("vol_20", 0.015), 1e-8)
    v60 = max(_safe_get("vol_60", 0.015), 1e-8)
    trend_s = _safe_get("trend_strength", 0.0)
    auto_c = _safe_get("autocorr_5", 0.0)
    vwap_d = _safe_get("vwap_dev", 0.0)
    rsi = _safe_get("rsi_14", 50.0)
    
    # Microstructure
    obi = (0.2 * _safe_get("obi_tau1") + 
           0.3 * _safe_get("obi_tau3") + 
           0.5 * _safe_get("obi_tau5"))
    spread = _safe_get("spread_bps", 5.0)
    depth_r = _safe_get("depth_ratio_5", 1.0)
    kyle_l = abs(_safe_get("kyle_lambda_est", 0.0))
    
    # Macro & Time
    funding = _safe_get("funding_rate", 0.0)
    is_us = bool(bar_context.get("is_us_session", False))
    is_wknd = bool(bar_context.get("is_weekend", False))
    min_to_fund = _safe_get("minutes_to_funding", 480.0)

    # 2. REGIME-BASED THRESHOLD CALCULATION
    # Base threshold adapted from high-performance seeds
    threshold_base = 0.00168 
    
    # Volatility Scaling Factor
    vol_impulse = v5 / v20
    vol_factor = 0.85 + (0.20 * min(2.0, vol_impulse))
    
    # Momentum scaling: lower threshold if trend is strong and price is persistent (autocorr)
    trend_factor = 1.0 - (0.05 * min(1.0, trend_s)) - (0.03 * max(0, min(1.0, auto_c)))
    
    # Weekend liquidity penalty
    weekend_multiplier = 1.15 if is_wknd else 1.0
    
    current_threshold = threshold_base * vol_factor * trend_factor * weekend_multiplier

    # 3. SIGNAL GENERATION & FILTERS
    signal = 0
    conviction = 0.0

    if predicted_return > current_threshold:
        # Long Filter: Require positive OBI, reasonable spread, and avoid VWAP overextension
        if obi > -0.75 and spread < 11.0 and vwap_d < 0.018:
            # Avoid longing if funding cost is excessive right before pay-out
            if not (funding > 0.002 and min_to_fund < 30):
                signal = 1
    elif predicted_return < -current_threshold:
        # Short Filter: Require negative OBI, reasonable spread, and avoid VWAP bottom-out
        if obi < 0.75 and spread < 11.0 and vwap_d > -0.018:
            # Avoid shorting if funding cost is excessive right before pay-out
            if not (funding < -0.002 and min_to_fund < 30):
                signal = -1

    if signal != 0:
        conviction = abs(predicted_return) / current_threshold

    # 4. POSITION SIZING
    if signal == 0:
        pos_size = 0.0
    else:
        # Base size calibration
        base_size = 0.150
        
        # Conviction boost
        conv_bonus = min(0.03, (conviction - 1.0) * 0.05) if conviction > 1.0 else 0.0
        
        # Session liquidity adjustment
        session_bonus = 0.01 if is_us else -0.005
        
        # Scale down if Kyle's Lambda is high (high price impact/low liquidity)
        impact_scalar = 1.0 - min(0.2, kyle_l * 100)
        
        pos_size = (base_size + conv_bonus + session_bonus) * impact_scalar
        pos_size = max(0.08, min(0.18, pos_size))

    # 5. DYNAMIC EXIT STRATEGY
    vol_trend = v5 / v60
    
    # TP/SL Elasticity scaled by long-term vol trend
    tp_mult = 0.95 + (0.05 * min(2.0, vol_trend))
    sl_mult = 0.98 + (0.04 * min(2.0, vol_impulse))
    
    take_profit = 0.0042 * tp_mult
    stop_loss = 0.0029 * sl_mult

    # RSI-based exhaustion adjustment
    if signal == 1 and rsi > 70:
        take_profit *= 0.9 # Take profits earlier if overbought
    elif signal == -1 and rsi < 30:
        take_profit *= 0.9 # Take profits earlier if oversold

    # 6. TIME EXIT
    # Standard 1-hour exit, reduced if market is stagnant
    max_bars = 4
    if vol_trend < 0.75 and trend_s < 0.3:
        max_bars = 3
    elif conviction > 1.4 and trend_s > 0.7:
        max_bars = 5 # Ride strong convictions slightly longer

    return {
        "signal": int(signal),
        "position_size": float(pos_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars)
    }
# EVOLVE-BLOCK-END
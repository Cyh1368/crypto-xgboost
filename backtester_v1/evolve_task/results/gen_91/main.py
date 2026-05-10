import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Advanced signal generator combining session-specific hurdles, liquidity-scaled 
    entry logic, and trend-aligned asymmetric risk management.
    """
    # 1. Feature Extraction
    obi = bar_context.get('obi_tau5', 0.0)
    pressure = bar_context.get('book_pressure_3', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    
    vol_5 = bar_context.get('vol_5', 0.001)
    vol_20 = bar_context.get('vol_20', 0.001)
    bb_pct = bar_context.get('bb_pct', 0.5)
    trend = bar_context.get('trend_strength', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    
    funding = bar_context.get('funding_rate', 0.0)
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    is_weekend = bar_context.get('is_weekend', False)
    vwap_dev = bar_context.get('vwap_dev', 0.0)

    # 2. Dynamic Entry Hurdle Calculation
    # Base hurdles vary by session characteristics
    if is_us:
        base_hurdle = 0.00168
    elif is_asia:
        base_hurdle = 0.00195
    else:
        base_hurdle = 0.00182

    # Liquidity penalty: Increase hurdle in illiquid conditions
    effective_hurdle = base_hurdle + min(0.0003, kyle_lambda * 30.0)
    
    # Weekend penalty: Liquidity is usually thinner
    if is_weekend:
        effective_hurdle += 0.0002
        
    # Volatility dampener: If short-term vol is spiking relative to long-term
    if vol_5 > vol_20 * 1.25:
        effective_hurdle *= 1.1
        
    # Trend alignment: Be more aggressive when prediction matches trend
    if (predicted_return > 0 and trend > 0.3) or (predicted_return < 0 and trend < -0.3):
        effective_hurdle *= 0.94

    # 3. Micro-Alignment & Signal Logic
    # Combine OBI and Book Pressure for liquidity confirmation
    micro_score = (obi + (pressure / 100.0)) / 2.0
    
    signal = 0
    if predicted_return > effective_hurdle:
        # Long Filter: Micro-alignment, spread check, and avoid buying extreme tops
        if micro_score > -0.45 and spread < 9.0 and vwap_dev < 0.022:
            if bb_pct < 0.88 and funding < 0.0018:
                signal = 1
                
    elif predicted_return < -effective_hurdle:
        # Short Filter: Micro-alignment, spread check, and avoid shorting extreme bottoms
        if micro_score < 0.45 and spread < 9.0 and vwap_dev > -0.022:
            if bb_pct > 0.12 and funding > -0.0018:
                signal = -1

    # 4. Adaptive Position Sizing
    if signal != 0:
        # Base size 0.155, scaled by confidence and liquidity impact
        confidence = abs(predicted_return) / effective_hurdle
        liquidity_scaler = max(0.75, 1.0 - (kyle_lambda * 350.0))
        
        position_size = 0.155 * min(1.25, confidence) * liquidity_scaler
        position_size = max(0.10, min(0.185, position_size))
    else:
        position_size = 0.0

    # 5. Risk Management (TP/SL)
    # Optimized TP/SL targets with asymmetric funding alignment
    tp = 0.0044
    sl = 0.0032
    
    if signal == 1:
        if funding < 0: # Longing while getting paid funding
            tp += 0.0005
            sl += 0.0001
    elif signal == -1:
        if funding > 0: # Shorting while getting paid funding
            tp += 0.0005
            sl += 0.0001

    # 6. Dynamic Hold Time
    # 5 bars for sessions/conditions with momentum, else 4
    if signal != 0:
        if is_us or abs(trend) > 0.35 or abs(autocorr) > 0.12:
            max_bars = 5
        else:
            max_bars = 4
    else:
        max_bars = 0
        tp = 0.0
        sl = 0.0

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(round(tp, 5)),
        "stop_loss": float(round(sl, 5)),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END
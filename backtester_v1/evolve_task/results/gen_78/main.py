import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Advanced signal generator using micro-alignment (OBI + Pressure), 
    range-relative hurdles, and liquidity-scaled sizing.
    """
    # 1. Feature Extraction
    close = bar_context.get('close', 1.0)
    obi = bar_context.get('obi_tau5', 0.0)
    pressure = bar_context.get('book_pressure_3', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    
    bb_pct = bar_context.get('bb_pct', 0.5)
    close_rank = bar_context.get('close_rank_48', 0.5)
    trend = bar_context.get('trend_strength', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    
    funding = bar_context.get('funding_rate', 0.0)
    is_us = bar_context.get('is_us_session', False)
    
    # 2. Dynamic Entry Hurdle
    # Base hurdle around the high-performing 0.0018 mark
    base_hurdle = 0.00185
    
    # Adjust hurdle based on range positioning to avoid over-extended entries
    long_hurdle = base_hurdle
    short_hurdle = base_hurdle
    
    if close_rank > 0.8 or bb_pct > 0.85:
        long_hurdle += 0.0003  # Harder to go long at the top
    if close_rank < 0.2 or bb_pct < 0.15:
        short_hurdle += 0.0003 # Harder to go short at the bottom

    # 3. Micro-Alignment Filter
    # Combine OBI and Book Pressure for a robust liquidity confirmation
    micro_score = (obi + (pressure / 100.0)) / 2.0 if 'book_pressure_3' in bar_context else obi
    
    signal = 0
    if predicted_return > long_hurdle:
        # Long filter: Positive prediction + OBI/Pressure not heavily against us
        if micro_score > -0.5 and spread < 10.0:
            # Avoid longing into extreme funding
            if funding < 0.0020:
                signal = 1
                
    elif predicted_return < -short_hurdle:
        # Short filter: Negative prediction + OBI/Pressure not heavily against us
        if micro_score < 0.5 and spread < 10.0:
            # Avoid shorting into extreme negative funding
            if funding > -0.0020:
                signal = -1

    # 4. Adaptive Position Sizing
    # Base size 0.135, scaled down by liquidity impact (Kyle's Lambda)
    if signal != 0:
        # kyle_lambda_est typically ranges from 0.00001 to 0.0005 in this context
        liquidity_scaler = max(0.7, 1.0 - (kyle_lambda * 400.0))
        position_size = 0.135 * liquidity_scaler
        # Confidence boost if prediction is significantly above hurdle
        confidence = abs(predicted_return) / base_hurdle
        position_size *= min(1.2, confidence)
        position_size = max(0.09, min(0.16, position_size))
    else:
        position_size = 0.0

    # 5. Risk Management (TP/SL)
    # Asymmetric TP/SL based on funding alignment
    tp = 0.0041
    sl = 0.0029
    
    if signal == 1:
        if funding < 0: # Getting paid to be long
            tp += 0.0004
            sl += 0.0001
    elif signal == -1:
        if funding > 0: # Getting paid to be short
            tp += 0.0004
            sl += 0.0001

    # 6. Dynamic Hold Time
    # Base 4 bars, extend to 5 if momentum is persistent or in US session
    if signal != 0:
        if is_us or autocorr > 0.1 or abs(trend) > 0.4:
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
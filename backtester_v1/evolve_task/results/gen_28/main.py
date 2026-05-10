import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Advanced signal generator using microstructure filters, volatility-regime 
    adjustments, and session-based hold logic.
    """
    # 1. Extract Contextual Features
    close = bar_context.get('close', 1.0)
    atr = bar_context.get('atr_14', 0.002)
    vol_ratio = bar_context.get('realized_vol_ratio', 1.0)
    spread = bar_context.get('spread_bps', 1.0)
    obi = bar_context.get('obi_tau5', 0.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    rsi = bar_context.get('rsi_14', 50.0)
    
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Dynamic Entry Threshold
    # Base threshold slightly lower than seed to capture more opportunities,
    # but filtered by spread and trend.
    base_thresh = 0.0019
    if abs(trend) > 0.6:
        base_thresh *= 0.9  # More aggressive in strong trends
    
    # Add a small penalty for high spreads
    entry_threshold = base_thresh + (min(spread, 10.0) * 0.00001)

    # 3. Signal Logic with Microstructure Filters
    signal = 0
    if predicted_return > entry_threshold:
        # Long Filter: Align with OBI, avoid extreme RSI, and check spread
        if obi > -0.4 and rsi < 72 and spread < 7.5:
            # Avoid longing if funding is extremely high (costly)
            if funding < 0.0008:
                signal = 1
    elif predicted_return < -entry_threshold:
        # Short Filter: Align with OBI, avoid extreme RSI, and check spread
        if obi < 0.4 and rsi > 28 and spread < 7.5:
            # Avoid shorting if funding is extremely negative (costly)
            if funding > -0.0008:
                signal = -1

    # 4. Adaptive Risk Management (TP/SL)
    # Scale targets by volatility ratio to adapt to market regime
    if signal != 0:
        # Base targets derived from the successful 0.004/0.003 seed
        vol_mult = max(0.85, min(1.25, vol_ratio))
        
        # Asymmetric TP/SL: Shorts often move faster/further in crypto
        if signal == 1:
            take_profit = 0.0041 * vol_mult
            stop_loss = 0.0031 * vol_mult
        else:
            take_profit = 0.0043 * vol_mult
            stop_loss = 0.0032 * vol_mult
            
        # 5. Dynamic Position Sizing
        # Base size 0.12, scaled by prediction conviction and liquidity
        confidence = abs(predicted_return) / 0.002
        # Reduce size if market impact (Kyle's Lambda) is high
        liquidity_scale = 1.0 / (1.0 + max(0, kyle_lambda * 500))
        
        position_size = 0.12 * min(1.2, confidence) * max(0.7, liquidity_scale)
        position_size = max(0.07, min(0.18, position_size))
        
        # 6. Optimized Hold Time (Max Bars)
        # US sessions trend more; Asia sessions mean-revert.
        if is_us:
            max_bars = 5 if autocorr > 0.1 else 4
        elif is_asia:
            max_bars = 4 if autocorr > 0.2 else 3
        else:
            max_bars = 4
            
        # Extend hold time slightly if trend is very strong
        if abs(trend) > 0.75:
            max_bars += 1
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

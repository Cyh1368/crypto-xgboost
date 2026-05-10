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
    # Lowered threshold to increase trade frequency, using microstructure as the filter.
    base_thresh = 0.00175
    signal = 0
    
    if predicted_return > base_thresh:
        # Long Filter: Confirm with OBI, ensure not over-leveraged/crowded (Funding)
        # Avoid entering if spread is prohibitive or price is too far above VWAP
        if avg_obi > -0.6 and funding < 0.0012 and spread < 11.0:
            if vwap_dev < 0.022:
                signal = 1
                
    elif predicted_return < -base_thresh:
        # Short Filter: Confirm with OBI, ensure funding isn't too negative
        if avg_obi < 0.6 and funding > -0.0012 and spread < 11.0:
            if vwap_dev > -0.022:
                signal = -1

    # 3. Position Sizing
    # High base size for high win-rate signals, minor penalty for high price impact (lambda)
    if signal != 0:
        base_size = 0.14
        # Confidence boost
        confidence_factor = min(1.3, abs(predicted_return) / base_thresh)
        # Liquidity penalty (Kyle's Lambda is return per unit volume)
        liquidity_adj = 1.0 - min(0.3, kyle_lambda * 1000)
        
        position_size = base_size * confidence_factor * liquidity_adj
        position_size = max(0.06, min(0.22, position_size))
        
        # 4. Target Setting (TP / SL)
        # Reverting to robust fixed targets with minor adjustments for trend alignment.
        # Historical performance shows 0.004 / 0.003 is the sweet spot for this model.
        tp_base = 0.0042
        sl_base = 0.0031
        
        # If we are trading with the trend, extend the take profit slightly
        if (signal == 1 and trend > 0.4) or (signal == -1 and trend < -0.4):
            tp_base *= 1.15
            sl_base *= 0.95
            
        take_profit = tp_base
        stop_loss = sl_base
        
        # 5. Time-Based Exit Strategy (Max Bars)
        # Use Autocorrelation to determine if we stay in a 'momentum' regime
        if is_us:
            # US session creates more persistence
            max_bars = 6 if autocorr > 0.05 else 5
        elif is_asia:
            # Asia session more likely to mean-revert
            max_bars = 4 if autocorr > 0.15 else 3
        else:
            max_bars = 4
            
        # Extension for strong trending moves
        if abs(trend) > 0.7:
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

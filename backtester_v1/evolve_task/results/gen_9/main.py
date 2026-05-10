import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Regime-adaptive signal generator that combines high-conviction thresholds 
    with microstructure liquidity filters and price-action guardrails.
    """
    
    # 1. Extract Core Features
    vol_5 = bar_context.get('vol_5', 0.002)
    vol_20 = bar_context.get('vol_20', 0.002)
    atr_14 = bar_context.get('atr_14', 0.001)
    close = bar_context.get('close', 1.0)
    obi = bar_context.get('obi_tau5', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    close_rank = bar_context.get('close_rank_48', 0.5)
    wick_up = bar_context.get('wick_ratio_up', 0.0)
    wick_down = bar_context.get('wick_ratio_down', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    
    # 2. Dynamic Thresholding
    # Base threshold inspired by Seed 3's 0.002, slightly adjusted for funding costs
    # If funding is positive, longs pay shorts; thus we require a higher return for longs.
    long_thresh = 0.00195 + max(0, funding * 0.5)
    short_thresh = -0.00195 + min(0, funding * 0.5)
    
    # 3. Signal Logic with Regime Filters
    signal = 0
    
    if predicted_return > long_thresh:
        # Long Filters: 
        # - Avoid buying if we are at the very top of the 48-bar range (Rank > 0.95)
        # - Avoid buying if price is overextended > 2% above VWAP
        # - Ensure OBI isn't heavily signaling selling pressure
        # - Avoid buying into a large upper wick (rejection)
        if close_rank < 0.95 and vwap_dev < 0.02 and obi > -0.4 and wick_up < 0.8:
            signal = 1
            
    elif predicted_return < short_thresh:
        # Short Filters:
        # - Avoid selling if we are at the very bottom of the 48-bar range (Rank < 0.05)
        # - Avoid selling if price is overextended > 2% below VWAP
        # - Ensure OBI isn't heavily signaling buying pressure
        # - Avoid selling into a large lower wick (rejection)
        if close_rank > 0.05 and vwap_dev > -0.02 and obi < 0.4 and wick_down < 0.8:
            signal = -1

    # 4. Position Sizing (Liquidity & Volatility Adjusted)
    # Base size 0.13, scaled down by spread and illiquidity (Kyle's Lambda)
    if signal != 0:
        # Scale down if spread is wide (> 6 bps) or Kyle's Lambda is high
        liquidity_mult = 1.0
        if spread > 6.0:
            liquidity_mult *= 0.8
        
        # Kyle's Lambda adjustment (penalize high impact environments)
        lambda_penalty = 1.0 / (1.0 + max(0, kyle_lambda) * 500)
        
        # Volatility scaling: reduce size if current vol is significantly higher than 20-bar avg
        vol_ratio = vol_5 / (vol_20 + 1e-9)
        vol_mult = 1.0 / max(1.0, vol_ratio)
        
        position_size = 0.13 * liquidity_mult * lambda_penalty * vol_mult
        position_size = min(0.20, max(0.06, position_size))
    else:
        position_size = 0.0

    # 5. Dynamic Exit Strategy (TP/SL)
    # Base TP/SL ratio ~1.35 (0.0042 / 0.0031)
    # Scale distances based on the vol_ratio to allow more room in high-vol regimes
    vol_expansion = 1.0
    if vol_5 > vol_20:
        vol_expansion = min(1.3, vol_5 / vol_20)
    
    # Use ATR as a secondary sanity check for TP/SL
    atr_pct = atr_14 / close if close > 0 else 0.001
    
    if signal != 0:
        # Base distances
        tp_dist = 0.0042 * vol_expansion
        sl_dist = 0.0031 * vol_expansion
        
        # Ensure SL is at least 1.2x ATR to avoid noise, but capped
        stop_loss = max(sl_dist, min(0.01, atr_pct * 1.2))
        take_profit = max(tp_dist, stop_loss * 1.3)
    else:
        take_profit = 0.0
        stop_loss = 0.0

    # 6. Time-Based Exit
    # Default to 4 bars (1 hour), but extend to 5 if trending (autocorr > 0.2)
    autocorr = bar_context.get('autocorr_5', 0.0)
    max_bars = 5 if abs(autocorr) > 0.2 else 4

    return {
        "signal": int(signal),
        "position_size": float(position_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END

import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Combines volatility-adaptive thresholding with liquiditiy impact filtering
    and vwap-relative conviction scaling.
    """
    import math
    import numpy as np

    def _get(key, default=0.0):
        v = bar_context.get(key, default)
        if v is None:
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    # 1. Feature Extraction
    # Microstructure
    obi1 = _get("obi_tau1")
    obi5 = _get("obi_tau5")
    book_pressure = _get("book_pressure_3")
    spread_bps = _get("spread_bps", 5.0)
    kyle_lambda = abs(_get("kyle_lambda_est", 0.0))
    
    # Price Action & Volatility
    vol5 = max(_get("vol_5", 0.015), 1e-8)
    vol20 = max(_get("vol_20", 0.015), 1e-8)
    vol60 = max(_get("vol_60", 0.015), 1e-8)
    vwap_dev = _get("vwap_dev")
    rsi14 = _get("rsi_14", 50.0)
    
    # Macro & Context
    funding = _get("funding_rate")
    is_us = bool(bar_context.get("is_us_session", False))
    
    # 2. Regime Classifiers
    # Volatility ratios
    vol_regime = vol5 / vol20
    vol_trend = vol5 / vol60
    
    # Microstructure Quality: Positive supports long, Negative supports short
    # Weights optimized for responsiveness to short-term book asymmetry
    micro_quality = (0.40 * obi1 + 0.20 * obi5 + 0.40 * book_pressure)
    
    # 3. Dynamic Thresholding
    # Base threshold sits at the high-performing 0.00171 level
    threshold_base = 0.00171
    # Multiplier expands the threshold during volatile spikes to filter noise
    elastic_multiplier = 0.87 + (0.21 * min(2.0, vol_regime))
    current_threshold = threshold_base * elastic_multiplier

    # 4. Signal Generation with Overextension Filter
    signal = 0
    signal_conviction = 0.0
    
    if predicted_return > current_threshold:
        # Long filter: check micro flow, funding cost, and RSI cap
        if micro_quality > -0.75 and funding < 0.0022 and spread_bps < 10.5 and rsi14 < 82:
            signal = 1
            signal_conviction = abs(predicted_return) / current_threshold
    elif predicted_return < -current_threshold:
        # Short filter: check micro flow, funding cost, and RSI floor
        if micro_quality < 0.75 and funding > -0.0022 and spread_bps < 10.5 and rsi14 > 18:
            signal = -1
            signal_conviction = abs(predicted_return) / current_threshold

    # 5. Position Sizing with Lambda and VWAP calibration
    if signal == 0:
        position_size = 0.0
    else:
        # Base size from high-sharpe foundations
        base_size = 0.148
        
        # Conviction boost scaled by relative intensity
        convic_factor = min(0.04, max(0, (signal_conviction - 1.0) * 0.07))
        
        # VWAP Adjustment: Boost size if entry is mean-reversion, penalize if overextended
        # (If signal is Long and vwap_dev is negative, we are buying 'cheap' relative to session)
        vwap_stretch = -vwap_dev * signal # Positive if mean-reverting
        vwap_mod = 1.0 + max(-0.1, min(0.1, vwap_stretch * 15.0))
        
        # Funding Carry Penalty
        funding_mod = 1.0
        if (signal == 1 and funding > 0.001) or (signal == -1 and funding < -0.001):
            funding_mod = 0.92
            
        # Kyle's Lambda Penalty: Reduce size if liquidity is thin/slippage is high
        liquidity_mod = 1.0
        if kyle_lambda > 0.00005: # Threshold for high slippage regime
            liquidity_mod = 0.85

        session_mod = 1.015 if is_us else 1.0
        
        position_size = (base_size + convic_factor) * vwap_mod * funding_mod * liquidity_mod * session_mod
        position_size = max(0.10, min(0.18, position_size))

    # 6. Exit Logic (ATR and Volatility calibrated)
    # Target higher risk-reward in trending environments
    tp_vol_adj = 0.94 + (0.07 * min(2.0, vol_trend))
    sl_vol_adj = 0.96 + (0.04 * min(2.2, vol_regime))
    
    take_profit = 0.0041 * tp_vol_adj
    stop_loss = 0.0031 * sl_vol_adj
    
    # 7. Time Exit
    max_bars = 4
    if vol_trend < 0.75 or vol_regime < 0.75:
        max_bars = 3 # Tighten time exit in stagnant regimes

    return {
        "signal":        int(signal),
        "position_size": float(position_size),
        "take_profit":   float(take_profit),
        "stop_loss":     float(stop_loss),
        "max_bars":      int(max_bars),
    }
# EVOLVE-BLOCK-END
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Seed trading signal generator.
    Args:
        predicted_return: float, model's predicted 15-min return (e.g. 0.003 = +0.3%)
        bar_context: dict with keys:
            - OHLCV: open, high, low, close, volume
            - Microstructure: obi_tau1, obi_tau3, obi_tau5, obi_tau10, spread_bps,
                              depth_ratio_5, depth_ratio_10, mid_price_move,
                              book_pressure_3, kyle_lambda_est
            - Price Action: ret_1, ret_3, ret_6, ret_12, ret_48, vol_5, vol_20, vol_60,
                            rsi_14, rsi_6, macd_signal, bb_pct, atr_14, momentum_bar,
                            wick_ratio_up, wick_ratio_down, volume_ratio_5, volume_ratio_20,
                            vwap_dev, autocorr_5, skew_20, kurt_20, realized_vol_ratio,
                            trend_strength, close_rank_48, gap_open, overnight_ret
            - Macro: funding_rate, funding_8h_ma
            - Time: hour_sin, hour_cos, dow_sin, dow_cos, is_asia_session,
                    is_us_session, is_weekend, minutes_to_funding
            - Aliases: atr (atr_14), rsi (rsi_14)
    Returns:
        dict with signal, position_size, take_profit, stop_loss, max_bars
    """
    # Base thresholds
    LONG_THRESH_BASE   = 0.0015   # lower base threshold for more sensitivity
    SHORT_THRESH_BASE  = -0.0015
    POSITION_SIZE_BASE = 0.10

    # Extract key indicators
    vol_20 = bar_context.get('vol_20', 0.01)
    atr_14 = bar_context.get('atr_14', bar_context.get('atr', 0.0))
    rsi_14 = bar_context.get('rsi_14', bar_context.get('rsi', 50.0))
    macd_signal = bar_context.get('macd_signal', 0.0)
    obi_tau5 = bar_context.get('obi_tau5', 0.0)
    is_us_session = bar_context.get('is_us_session', False)
    is_asia_session = bar_context.get('is_asia_session', False)
    trend_strength = bar_context.get('trend_strength', 0.0)
    momentum_bar = bar_context.get('momentum_bar', 0.0)

    # Normalize volatility (20-day vol as baseline)
    vol_multiplier = max(0.5, min(2.0, vol_20 / 0.012))  # 1.2% is neutral vol

    # Adjust thresholds based on session (US more volatile, need stronger signals)
    if is_us_session:
        long_thresh = LONG_THRESH_BASE * 1.2
        short_thresh = SHORT_THRESH_BASE * 1.2
    elif is_asia_session:
        long_thresh = LONG_THRESH_BASE * 0.9
        short_thresh = SHORT_THRESH_BASE * 0.9
    else:
        long_thresh = LONG_THRESH_BASE
        short_thresh = SHORT_THRESH_BASE

    # Momentum and RSI confirmation filters
    rsi_extreme_long = rsi_14 > 70.0    # overbought, reduce long signal strength
    rsi_extreme_short = rsi_14 < 30.0   # oversold, reduce short signal strength
    rsi_favorable_long = 40.0 < rsi_14 < 65.0  # neutral to bullish is favorable
    rsi_favorable_short = 35.0 < rsi_14 < 60.0  # neutral to bearish is favorable

    # Trend confirmation (MACD signal and momentum)
    macd_long_support = macd_signal > 0.0
    macd_short_support = macd_signal < 0.0
    momentum_long_support = momentum_bar > 0.0
    momentum_short_support = momentum_bar < 0.0

    # Order book imbalance confidence (absolute value, normalized)
    obi_confidence = min(1.0, abs(obi_tau5) / 0.1) if abs(obi_tau5) > 0.001 else 0.5

    # Entry logic with multi-factor confirmation
    signal = 0
    entry_confidence = 0.0

    if predicted_return > long_thresh:
        # Long signal with filters
        has_momentum = macd_long_support or momentum_long_support
        has_rsi_support = rsi_favorable_long and not rsi_extreme_long
        trend_aligned = trend_strength > 0.3

        # Require at least 2 of 3 confirmations (momentum, RSI, trend)
        confirmations = int(has_momentum) + int(has_rsi_support) + int(trend_aligned)
        if confirmations >= 2:
            signal = 1
            entry_confidence = 0.5 + 0.25 * confirmations / 3.0 + 0.25 * obi_confidence
            entry_confidence = min(1.0, entry_confidence)

    elif predicted_return < short_thresh:
        # Short signal with filters
        has_momentum = macd_short_support or momentum_short_support
        has_rsi_support = rsi_favorable_short and not rsi_extreme_short
        trend_aligned = trend_strength < -0.3

        # Require at least 2 of 3 confirmations (momentum, RSI, trend)
        confirmations = int(has_momentum) + int(has_rsi_support) + int(trend_aligned)
        if confirmations >= 2:
            signal = -1
            entry_confidence = 0.5 + 0.25 * confirmations / 3.0 + 0.25 * obi_confidence
            entry_confidence = min(1.0, entry_confidence)

    # Position sizing: scale with volatility and entry confidence
    # Lower volatility and higher confidence = larger position
    position_size = 0.0
    if signal != 0:
        base_pos = POSITION_SIZE_BASE * entry_confidence
        vol_adjusted_pos = base_pos / vol_multiplier
        position_size = max(0.02, min(0.15, vol_adjusted_pos))  # cap between 2% and 15%

    # Risk management: volatility-adjusted stops and profits
    # Use ATR as primary stop, vol-adjusted as secondary
    if atr_14 > 0:
        atr_multiple = 1.5 if entry_confidence < 0.6 else 1.0  # wider stops for low confidence
        stop_loss = atr_14 * atr_multiple
    else:
        stop_loss = 0.003

    # Take profit: asymmetric (reward better for high confidence, otherwise take less)
    base_tp = 0.004
    tp_multiplier = 0.8 + 0.4 * entry_confidence  # ranges from 0.8 to 1.2x
    take_profit = base_tp * tp_multiplier

    # Risk/reward ratio check: if RR < 1.5, skip the trade
    if signal != 0 and take_profit < stop_loss * 1.5:
        signal = 0
        position_size = 0.0

    # Session-adaptive max bars (longer holds in trending sessions)
    if is_us_session:
        max_bars = 6  # longer hold during US session (1.5 hours)
    elif trend_strength > 0.5:
        max_bars = 5
    else:
        max_bars = 4

    return {
        "signal":        signal,
        "position_size": position_size,
        "take_profit":   take_profit,
        "stop_loss":     stop_loss,
        "max_bars":      max_bars,
    }
# EVOLVE-BLOCK-END
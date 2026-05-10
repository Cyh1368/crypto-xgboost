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
    # Re-introducing session-specific thresholds to capture volatility profile
    if is_us:
        base_hurdle = 0.00171
    elif is_asia:
        base_hurdle = 0.00202
    else:
        base_hurdle = 0.00184

    # Adjust hurdle based on range positioning to avoid over-extended entries
    long_hurdle = base_hurdle
    short_hurdle = base_hurdle

    if close_rank > 0.85 or bb_pct > 0.9:
        long_hurdle *= 1.15  # Scale hurdle rather than flat addition
    if close_rank < 0.15 or bb_pct < 0.1:
        short_hurdle *= 1.15

    # 3. Micro-Alignment and VWAP Filters
    micro_score = (obi + (pressure / 100.0)) / 2.0 if 'book_pressure_3' in bar_context else obi
    vwap_dev = bar_context.get('vwap_dev', 0.0)

    signal = 0
    if predicted_return > long_hurdle:
        # Filter for liquidity alignment and avoid buying extreme VWAP extensions
        if micro_score > -0.55 and spread < 10.5 and vwap_dev < 0.024:
            if funding < 0.0016:
                signal = 1

    elif predicted_return < -short_hurdle:
        # Filter for liquidity alignment and avoid selling extreme VWAP extensions
        if micro_score < 0.55 and spread < 10.5 and vwap_dev > -0.024:
            if funding > -0.0016:
                signal = -1

    # 4. Adaptive Position Sizing
    # Increasing base size to 0.145 and cap to 0.19 to maximize return potential
    if signal != 0:
        liquidity_scaler = max(0.75, 1.0 - (kyle_lambda * 350.0))
        confidence = abs(predicted_return) / base_hurdle
        position_size = 0.145 * min(1.3, confidence) * liquidity_scaler

        # Volatility check: Scale down in outlier ATR environments
        atr_pct = (bar_context.get('atr_14', 0) / close) if close > 0 else 0.002
        if atr_pct > 0.012:
            position_size *= 0.82

        position_size = max(0.08, min(0.19, position_size))
    else:
        position_size = 0.0

    # 5. Risk Management (TP/SL)
    # Slightly wider targets for better coverage of 15m return distribution
    tp = 0.0042
    sl = 0.0031

    if (signal == 1 and funding < 0) or (signal == -1 and funding > 0):
        tp += 0.0005
        sl += 0.0001

    # Increase TP ceiling in strong trending states
    if abs(trend) > 0.4:
        tp += 0.0004

    # 6. Dynamic Hold Time
    # Asia session holds for less time, US/Trend holds for longer
    if signal != 0:
        if is_us or abs(trend) > 0.35:
            max_bars = 6 if abs(autocorr) > 0.15 else 5
        elif is_asia:
            max_bars = 4
        else:
            max_bars = 5
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
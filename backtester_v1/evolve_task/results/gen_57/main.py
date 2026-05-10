import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Regime-aware trading signal generator with asymmetric filters,
    volatility-adjusted risk, and microstructure-aware entry control.
    """
    # Core market state
    close = bar_context.get('close', 1.0)
    atr = bar_context.get('atr_14', bar_context.get('atr', 0.002))
    vol_5 = bar_context.get('vol_5', 0.001)
    vol_20 = bar_context.get('vol_20', 0.001)
    vol_60 = bar_context.get('vol_60', 0.001)
    rsi_14 = bar_context.get('rsi_14', bar_context.get('rsi', 50.0))
    rsi_6 = bar_context.get('rsi_6', 50.0)
    bb_pct = bar_context.get('bb_pct', 0.5)
    trend = bar_context.get('trend_strength', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    momentum = bar_context.get('momentum_bar', 0.0)
    vwap_dev = bar_context.get('vwap_dev', 0.0)
    wick_up = bar_context.get('wick_ratio_up', 0.0)
    wick_dn = bar_context.get('wick_ratio_down', 0.0)
    spread = bar_context.get('spread_bps', 1.0)
    obi3 = bar_context.get('obi_tau3', 0.0)
    obi5 = bar_context.get('obi_tau5', 0.0)
    depth5 = bar_context.get('depth_ratio_5', 1.0)
    pressure3 = bar_context.get('book_pressure_3', 0.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    funding = bar_context.get('funding_rate', 0.0)
    funding_ma = bar_context.get('funding_8h_ma', 0.0)
    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    is_weekend = bar_context.get('is_weekend', False)
    minutes_to_funding = bar_context.get('minutes_to_funding', 999)

    # Combine microstructure into a stable directional score
    obi = 0.5 * obi5 + 0.3 * obi3 + 0.2 * pressure3
    liquidity = max(0.2, min(2.5, depth5))
    vol_ref = max(1e-6, vol_20)

    # Regime-adjusted entry thresholds
    long_thresh = 0.0020
    short_thresh = -0.0020

    # Higher cost / toxic flow -> demand more edge
    long_thresh += min(0.0005, spread * 0.00003)
    short_thresh -= min(0.0005, spread * 0.00003)
    long_thresh += min(0.0004, kyle_lambda * 30.0)
    short_thresh -= min(0.0004, kyle_lambda * 30.0)

    # Session modulation
    if is_us:
        long_thresh -= 0.00012
        short_thresh += 0.00012
    elif is_asia:
        long_thresh += 0.00010
        short_thresh -= 0.00010

    # Weekend / funding caution
    if is_weekend:
        long_thresh += 0.00010
        short_thresh -= 0.00010
    if minutes_to_funding < 25:
        long_thresh += 0.00012
        short_thresh -= 0.00012

    # Volatility and mean-reversion regime
    if vol_5 > vol_20 * 1.15:
        long_thresh += 0.00010
        short_thresh -= 0.00010
    if abs(autocorr) > 0.20 or abs(trend) > 0.35:
        long_thresh -= 0.00008
        short_thresh += 0.00008

    # Signal selection with asymmetric filters
    signal = 0
    if predicted_return > long_thresh:
        long_ok = (
            obi > -0.15 and
            spread < 10.0 and
            vwap_dev < 0.018 and
            wick_up < 0.78 and
            bb_pct < 0.92 and
            (rsi_6 < 74.0 or rsi_14 < 68.0) and
            momentum >= -0.0005 and
            funding <= funding_ma + 0.00025
        )
        if long_ok:
            signal = 1
    elif predicted_return < short_thresh:
        short_ok = (
            obi < 0.15 and
            spread < 10.0 and
            vwap_dev > -0.018 and
            wick_dn < 0.78 and
            bb_pct > 0.08 and
            (rsi_6 > 26.0 or rsi_14 > 32.0) and
            momentum <= 0.0005 and
            funding >= funding_ma - 0.00025
        )
        if short_ok:
            signal = -1

    # Position sizing: confidence + liquidity + toxicity adjustment
    if signal != 0:
        edge = abs(predicted_return) / max(1e-6, abs(long_thresh) if signal == 1 else abs(short_thresh))
        confidence = max(0.75, min(1.35, edge))

        liq_penalty = 1.0
        if spread > 8.0:
            liq_penalty *= 0.90
        if kyle_lambda > 0.0005:
            liq_penalty *= 0.85
        if vol_20 > vol_60 * 1.10:
            liq_penalty *= 0.90
        if liquidity < 0.8:
            liq_penalty *= 0.92

        if signal == 1:
            base_size = 0.115
        else:
            base_size = 0.105

        position_size = base_size * confidence * liq_penalty
        position_size = max(0.06, min(0.18, position_size))
    else:
        position_size = 0.0

    # Volatility-adjusted exits with asymmetric logic
    if signal != 0:
        atr_pct = atr / close if close > 0 else 0.003

        # Base targets
        take_profit = max(0.0035, min(0.0125, atr_pct * 1.55))
        stop_loss = max(0.0026, min(0.0085, atr_pct * 1.08))

        # Trend-following trades can breathe more
        if abs(trend) > 0.35 and abs(autocorr) > 0.15:
            take_profit *= 1.12
            stop_loss *= 0.95

        # Mean-reversion / stretched conditions tighten targets
        if (signal == 1 and bb_pct > 0.85) or (signal == -1 and bb_pct < 0.15):
            take_profit *= 0.92
            stop_loss *= 0.90

        # Funding stress -> faster exits
        if abs(funding - funding_ma) > 0.00025:
            take_profit *= 0.96
            stop_loss *= 0.94
    else:
        take_profit = 0.0
        stop_loss = 0.0

    # Adaptive time stop
    if signal == 0:
        max_bars = 0
    elif is_us and (abs(trend) > 0.35 or abs(autocorr) > 0.15):
        max_bars = 5
    elif is_asia or is_weekend:
        max_bars = 3
    else:
        max_bars = 4

    return {
        "signal": int(signal),
        "position_size": float(round(position_size, 4)),
        "take_profit": float(round(take_profit, 5)),
        "stop_loss": float(round(stop_loss, 5)),
        "max_bars": int(max_bars),
    }
# EVOLVE-BLOCK-END
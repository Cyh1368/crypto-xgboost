import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Refined trading signal generator using liquidity-adjusted thresholds,
    autocorrelation-driven hold times, and volatility-normalized sizing.
    """
    # 1. Essential Market Markers
    close = bar_context.get('close', 1.0)
    vol_short = bar_context.get('vol_5', 0.001)
    vol_long = bar_context.get('vol_20', 0.001)
    atr = bar_context.get('atr_14', 0.002)
    vol_ratio = bar_context.get('realized_vol_ratio', 1.0)
    kyle_lambda = bar_context.get('kyle_lambda_est', 0.0)
    autocorr = bar_context.get('autocorr_5', 0.0)
    trend = bar_context.get('trend_strength', 0.0)
    rsi = bar_context.get('rsi_14', 50.0)

    is_us = bar_context.get('is_us_session', False)
    is_asia = bar_context.get('is_asia_session', False)
    funding = bar_context.get('funding_rate', 0.0)

    # 2. Dynamic Threshold Logic
    # Using a slightly lower threshold than the seed to capture more opportunities,
    # while using spread and OBI as light quality filters.
    spread_bps = bar_context.get('spread_bps', 1.0)
    obi = bar_context.get('obi_tau5', 0.0)
    base_thresh = 0.0017

    # 3. Execution Signal Logic
    signal = 0
    if predicted_return > base_thresh and spread_bps < 6.0:
        # Loosened filters to increase trade frequency while avoiding extremes
        if rsi < 85 and funding < 0.0008 and obi > -0.5:
            signal = 1
    elif predicted_return < -base_thresh and spread_bps < 6.0:
        if rsi > 15 and funding > -0.0008 and obi < 0.5:
            signal = -1

    # 4. Adaptive Hold Time (Max Bars)
    # Standardizing on the seed's 4-bar hold (1 hour), with a slight extension for strong trends.
    max_bars = 4
    if abs(trend) > 0.6 or (is_us and autocorr > 0.1):
        max_bars = 5

    # 5. Dynamic Risk Targets (TP/SL)
    # Reverting toward the seed's high-performing fixed TP/SL (0.004/0.003)
    # with minimal ATR-based scaling to handle volatility spikes.
    if signal != 0:
        atr_pct = (atr / close) if close > 0 else 0.003

        # Keep TP/SL tight and close to the 1.33 ratio to maximize Sharpe/Calmar
        take_profit = max(0.0042, min(0.008, atr_pct * 1.4))
        stop_loss = max(0.0032, min(0.006, atr_pct * 1.0))

        # 6. Position Sizing
        # Stable sizing near 0.11, with minor adjustments for conviction.
        # Capping size at 0.13 to reduce Max Drawdown.
        confidence = min(1.3, abs(predicted_return) / 0.0017)
        position_size = 0.11 * confidence
        position_size = max(0.07, min(0.13, position_size))
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
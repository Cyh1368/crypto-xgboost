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
    LONG_THRESH   = 0.002    # enter long if predicted return > 0.2%
    SHORT_THRESH  = -0.002   # enter short if predicted return < -0.2%
    POSITION_SIZE = 0.10     # 10% of portfolio per trade
    TAKE_PROFIT   = 0.004    # exit at +0.4% gain
    STOP_LOSS     = 0.003    # exit at -0.3% loss
    MAX_BARS      = 4        # time-based exit after 4 bars (1 hour)

    if predicted_return > LONG_THRESH:
        signal = 1
    elif predicted_return < SHORT_THRESH:
        signal = -1
    else:
        signal = 0

    return {
        "signal":        signal,
        "position_size": POSITION_SIZE if signal != 0 else 0.0,
        "take_profit":   TAKE_PROFIT,
        "stop_loss":     STOP_LOSS,
        "max_bars":      MAX_BARS,
    }
# EVOLVE-BLOCK-END

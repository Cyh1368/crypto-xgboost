import random
import numpy as np

random.seed(42)
np.random.seed(42)

# EVOLVE-BLOCK-START
def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Seed trading signal generator.
    Translates XGBoost 15-min return prediction into a trading decision.
    Args:
        predicted_return: float, model's predicted 15-min return (e.g. 0.003 = +0.3%)
        bar_context: dict with keys: close, high, low, volume, atr, rsi, adx
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

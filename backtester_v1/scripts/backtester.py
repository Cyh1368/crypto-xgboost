import xgboost as xgb
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# Constants from Guide
BPS_TO_RATIO = 1 / 10000
TAKER_FEE = 0.0005 # 5 BPS
POSITION_SIZE_USD = 10000
MAX_LEVERAGE = 3.0

@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    qty_btc: float
    pnl_usd: float
    fees_usd: float
    net_pnl_usd: float

@dataclass
class BacktestState:
    equity: float = 100000.0
    position: int = 0
    entry_price: float = 0.0
    entry_ts: pd.Timestamp = None
    qty_btc: float = 0.0
    bars_held: int = 0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    max_bars: int = 0
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

def load_model(model_path: str):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def predict_ratios(model: xgb.Booster, features_df: pd.DataFrame, calibration_factor: float = 1.0) -> np.ndarray:
    dmatrix = xgb.DMatrix(features_df)
    bps_raw = model.predict(dmatrix)
    bps_cal = bps_raw * calibration_factor
    ratios = 1.0 + (bps_cal * BPS_TO_RATIO)
    return ratios

def generate_signal(predicted_return: float, bar_context: dict) -> dict:
    """
    Best strategy found by shinkaevolve.
    """
    def _safe_get(key, default=0.0, cast_fn=float):
        v = bar_context.get(key, default)
        if v is None: return default
        try: return cast_fn(v)
        except (ValueError, TypeError): return default

    # Microstructure parameters
    obi_tau1 = _safe_get("obi_tau1", 0.0)
    obi_tau3 = _safe_get("obi_tau3", 0.0)
    obi_tau5 = _safe_get("obi_tau5", 0.0)
    spread_bps = _safe_get("spread_bps", 5.0)
    book_pressure = _safe_get("book_pressure_3", 0.0)
    
    # Price action & volatility
    vol_5 = max(_safe_get("vol_5", 0.015), 1e-8)
    vol_20 = max(_safe_get("vol_20", 0.015), 1e-8)
    vol_60 = max(_safe_get("vol_60", 0.015), 1e-8)
    trend_strength = _safe_get("trend_strength", 0.0)

    # Macro parameters
    funding_rate = _safe_get("funding_rate", 0.0)

    # Session/time parameters
    is_us_session = bar_context.get("is_us_session", False) or False

    # Signal Classification
    vol_impulse = vol_5 / vol_20
    vol_impulse = max(0.5, min(2.0, vol_impulse))

    THRESHOLD_BASE = 0.00172
    elastic_multiplier = 0.88 + (0.18 * min(2.0, vol_impulse))
    trend_adj = 1.0 - (0.04 * min(1.0, trend_strength))
    current_threshold = THRESHOLD_BASE * elastic_multiplier * trend_adj

    signal = 0
    signal_conviction = 0.0

    if predicted_return > current_threshold:
        signal = 1
        signal_conviction = predicted_return / current_threshold
    elif predicted_return < -current_threshold:
        signal = -1
        signal_conviction = abs(predicted_return) / current_threshold

    # Quality Filters
    if signal != 0:
        micro_quality = (0.32 * obi_tau1 + 0.24 * obi_tau3 + 0.24 * obi_tau5 + 0.20 * book_pressure)
        if signal == 1:
            if not (micro_quality > -0.85 and funding_rate < 0.0025 and spread_bps < 10.5):
                signal = 0
        elif signal == -1:
            if not (micro_quality < 0.85 and funding_rate > -0.0025 and spread_bps < 10.5):
                signal = 0

    # Risk Management
    if signal == 0:
        position_size = 0.0
    else:
        base_size = 0.145
        conviction_bonus = min(0.035, max(0, (signal_conviction - 1.0) * 0.06))
        session_bonus = 0.01 if is_us_session else 0.0
        funding_filter = 1.0
        if (signal == 1 and funding_rate > 0.0015) or (signal == -1 and funding_rate < -0.0015):
            funding_filter = 0.90
        position_size = (base_size + conviction_bonus + session_bonus) * funding_filter
        position_size = max(0.10, min(0.18, position_size))

    vol_trend = vol_5 / vol_60
    vol_trend = max(0.5, min(2.0, vol_trend))
    tp_elasticity = 0.94 + (0.06 * min(2.0, vol_trend))
    sl_elasticity = 0.97 + (0.03 * min(2.0, vol_impulse))

    take_profit = 0.0040 * tp_elasticity
    stop_loss = 0.0030 * sl_elasticity
    max_bars = 4
    if vol_trend < 0.8: max_bars = 3

    return {
        "signal": int(signal),
        "position_size": float(position_size),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_bars": int(max_bars),
    }

def run_backtest(ohlcv_df: pd.DataFrame, features_df: pd.DataFrame, model: xgb.Booster, initial_equity: float = 100000.0, calibration_factor: float = 1.0) -> BacktestState:
    state = BacktestState(equity=initial_equity)
    predicted_ratios = predict_ratios(model, features_df, calibration_factor)
    
    for i, (ts, feat_row) in enumerate(features_df.iterrows()):
        price_now = ohlcv_df.loc[ts, 'close']
        pred_ret = predicted_ratios[i] - 1.0
        
        # Signal Generation
        bar_context = feat_row.to_dict()
        bar_context['open'] = ohlcv_df.loc[ts, 'open']
        bar_context['high'] = ohlcv_df.loc[ts, 'high']
        bar_context['low'] = ohlcv_df.loc[ts, 'low']
        bar_context['close'] = price_now
        bar_context['volume'] = ohlcv_df.loc[ts, 'volume']
        
        out = generate_signal(pred_ret, bar_context)
        signal = out['signal']
        
        # Check Exits for existing position
        exit_reason = None
        if state.position != 0:
            state.bars_held += 1
            pnl_pct = (price_now - state.entry_price) / state.entry_price * state.position
            
            if pnl_pct >= state.take_profit:
                exit_reason = "TP"
            elif pnl_pct <= -state.stop_loss:
                exit_reason = "SL"
            elif state.bars_held >= state.max_bars:
                exit_reason = "TIME"
            elif signal != 0 and signal != state.position:
                exit_reason = "FLIP"
            elif signal == 0:
                exit_reason = "SIGNAL"

        # Close position
        if exit_reason:
            exit_price = price_now
            pnl = state.qty_btc * (exit_price - state.entry_price) * state.position
            fees = (state.qty_btc * state.entry_price + state.qty_btc * exit_price) * TAKER_FEE
            net_pnl = pnl - fees
            state.equity += net_pnl
            
            state.trades.append(Trade(
                entry_ts=state.entry_ts,
                exit_ts=ts,
                direction=state.position,
                entry_price=state.entry_price,
                exit_price=exit_price,
                qty_btc=state.qty_btc,
                pnl_usd=pnl,
                fees_usd=fees,
                net_pnl_usd=net_pnl
            ))
            state.position = 0
            state.qty_btc = 0.0
            state.bars_held = 0

        # Open new position
        if signal != 0 and state.position == 0:
            state.position = signal
            state.entry_price = price_now
            state.entry_ts = ts
            state.bars_held = 0
            state.take_profit = out['take_profit']
            state.stop_loss = out['stop_loss']
            state.max_bars = out['max_bars']
            
            # Position sizing
            notional = state.equity * out['position_size']
            # Cap by leverage
            max_notional = state.equity * MAX_LEVERAGE
            notional = min(notional, max_notional)
            state.qty_btc = notional / price_now
            
        # Equity Curve Tracking (MTM)
        unrealized = 0
        if state.position != 0:
            unrealized = state.qty_btc * (price_now - state.entry_price) * state.position
            
        state.equity_curve.append(state.equity + unrealized)
        state.timestamps.append(ts)
        state.positions.append(state.position)
        
    return state

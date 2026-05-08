import xgboost as xgb
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# Constants from Guide
CALIBRATION_FACTOR = 1.0411
BPS_TO_RATIO = 1 / 10000
TAKER_FEE = 0.0005 # 5 BPS
POSITION_SIZE_USD = 10000
MAX_LEVERAGE = 3.0

# Strategy Thresholds (Start with default from guide)
LONG_THRESHOLD = 1.000080
SHORT_THRESHOLD = 0.999920

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
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

def load_model(model_path: str):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def predict_ratios(model: xgb.Booster, features_df: pd.DataFrame) -> np.ndarray:
    dmatrix = xgb.DMatrix(features_df)
    bps_raw = model.predict(dmatrix)
    bps_cal = bps_raw * CALIBRATION_FACTOR
    ratios = 1.0 + (bps_cal * BPS_TO_RATIO)
    return ratios

def apply_risk_filters(signal: int, features_row: pd.Series) -> int:
    if features_row['spread_bps'] > 10:
        return 0
    if features_row['vol_5'] > 3 * features_row['vol_60']:
        return 0
    
    fr = features_row['funding_rate']
    if signal == 1 and fr < -0.001:
        return 0
    if signal == -1 and fr > 0.001:
        return 0
    return signal

def run_backtest(ohlcv_df: pd.DataFrame, features_df: pd.DataFrame, model: xgb.Booster, initial_equity: float = 100000.0) -> BacktestState:
    state = BacktestState(equity=initial_equity)
    predicted_ratios = predict_ratios(model, features_df)
    
    for i, (ts, feat_row) in enumerate(features_df.iterrows()):
        price_now = ohlcv_df.loc[ts, 'close']
        pred_ratio = predicted_ratios[i]
        
        # Signal Generation
        raw_signal = 0
        if pred_ratio > LONG_THRESHOLD:
            raw_signal = 1
        elif pred_ratio < SHORT_THRESHOLD:
            raw_signal = -1
            
        signal = apply_risk_filters(raw_signal, feat_row)
        
        # Close existing position if signal flips or goes flat
        if state.position != 0 and signal != state.position:
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
            
        # Open new position
        if signal != 0 and state.position == 0:
            state.position = signal
            state.entry_price = price_now
            state.entry_ts = ts
            
            max_notional = state.equity * MAX_LEVERAGE
            notional = min(POSITION_SIZE_USD, max_notional)
            state.qty_btc = notional / price_now
            
        # Equity Curve Tracking (MTM)
        unrealized = 0
        if state.position != 0:
            unrealized = state.qty_btc * (price_now - state.entry_price) * state.position
            
        state.equity_curve.append(state.equity + unrealized)
        state.timestamps.append(ts)
        
    return state

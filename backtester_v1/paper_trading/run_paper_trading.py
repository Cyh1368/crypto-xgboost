import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import ccxt
from datetime import datetime, timedelta

# Ensure scripts are importable
sys.path.append(os.getcwd())

from backtester_v1.scripts.feature_engineering import build_features, FEATURE_NAMES
from backtester_v1.scripts.backtester import generate_signal

# --- CONFIGURATION ---
# Mapping Binance-style symbols to Kraken Futures symbols
SYMBOL_MAP = {
    'BTC/USDT':  'BTC/USD:USD',
    'ETH/USDT':  'ETH/USD:USD',
    'SOL/USDT':  'SOL/USD:USD',
    'BNB/USDT':  'BNB/USD:USD',
    'XRP/USDT':  'XRP/USD:USD',
    'ADA/USDT':  'ADA/USD:USD',
    'DOGE/USDT': 'DOGE/USD:USD',
    'DOT/USDT':  'DOT/USD:USD',
    'LINK/USDT': 'LINK/USD:USD',
    'LTC/USDT':  'LTC/USD:USD'
}
SYMBOLS = list(SYMBOL_MAP.keys())
BASE_DIR = 'backtester_v1/paper_trading/results'
MODEL_PATH = 'backtester_v1/models/xgb_regression_v1.json'
SCALER_PATH = 'backtester_v1/models/scaler_v1.joblib'
CALIB_PATH = 'backtester_v1/models/calibration_v1.joblib'
INITIAL_TOTAL_EQUITY = 100.0
EQUITY_PER_COIN = 10.0
TIMEFRAME = '15m'
FEE_RATE = 0.0005 # 5 BPS

# --- LOGGING SETUP ---
os.makedirs(BASE_DIR, exist_ok=True)
log_file = os.path.join(BASE_DIR, 'paper_trading.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('PaperTrader')

class PaperTrader:
    def __init__(self):
        # Switching to Kraken Futures due to Binance regional restrictions
        self.exchange = ccxt.krakenfutures({'enableRateLimit': True})
        self.model = xgb.Booster()
        self.model.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.calibration_factor = joblib.load(CALIB_PATH)
        
        # State per symbol
        self.states = {}
        for symbol in SYMBOLS:
            self.states[symbol] = {
                'position': 0, 
                'entry_price': 0.0,
                'qty': 0.0,
                'equity': EQUITY_PER_COIN,
                'bars_held': 0,
                'take_profit': 0.0,
                'stop_loss': 0.0,
                'max_bars': 0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0
            }
        
        self.total_pnl = 0.0
        self.total_equity = INITIAL_TOTAL_EQUITY
        self._init_csvs()

    def _init_csvs(self):
        for symbol in SYMBOLS:
            safe_name = symbol.replace('/', '_')
            path = os.path.join(BASE_DIR, f'trades_{safe_name}.csv')
            if not os.path.exists(path):
                df = pd.DataFrame(columns=['timestamp', 'price', 'prediction', 'action', 'position', 'fee', 'pnl'])
                df.to_csv(path, index=False)

        path = os.path.join(BASE_DIR, 'total_pnl.csv')
        if not os.path.exists(path):
            df = pd.DataFrame(columns=['timestamp', 'total_equity', 'total_pnl'])
            df.to_csv(path, index=False)

    def fetch_data(self, b_symbol):
        """Fetch historical OHLCV and current OB/Funding from Kraken with retries."""
        k_symbol = SYMBOL_MAP[b_symbol]
        max_retries = 3
        retry_delay = 2 # seconds
        
        for attempt in range(max_retries):
            try:
                # Fetch last 200 bars for EMA convergence
                ohlcv = self.exchange.fetch_ohlcv(k_symbol, timeframe=TIMEFRAME, limit=200)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Get snapshots
                ob = self.exchange.fetch_order_book(k_symbol, limit=20)
                funding = self.exchange.fetch_funding_rate(k_symbol)
                
                # Inject current microstructure into the latest bar
                last_idx = df.index[-1]
                df = df.copy() # Avoid fragmentation
                df['bids'] = None
                df['asks'] = None
                df['funding_rate'] = None
                
                # Initialize with current values
                df.at[last_idx, 'bids'] = ob['bids']
                df.at[last_idx, 'asks'] = ob['asks']
                df.at[last_idx, 'funding_rate'] = funding['fundingRate']
                
                # Fill backward to allow indicator calculation
                df['bids'] = df['bids'].ffill().bfill()
                df['asks'] = df['asks'].ffill().bfill()
                df['funding_rate'] = df['funding_rate'].ffill().bfill()
                
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {b_symbol} after error: {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error fetching data for {b_symbol} ({k_symbol}) after {max_retries} attempts: {e}")
        return None

    def predict(self, df):
        try:
            features = build_features(df, scaler=self.scaler)
            if features.empty:
                return None, None
            
            latest_feat = features.tail(1)
            dmatrix = xgb.DMatrix(latest_feat)
            bps_raw = self.model.predict(dmatrix)[0]
            bps_cal = bps_raw * self.calibration_factor
            pred_ret = bps_cal / 10000.0
            
            bar_context = latest_feat.iloc[0].to_dict()
            raw_data = df.loc[latest_feat.index[0]]
            bar_context.update({
                'open': raw_data['open'],
                'high': raw_data['high'],
                'low': raw_data['low'],
                'close': raw_data['close'],
                'volume': raw_data['volume']
            })
            
            return pred_ret, bar_context
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, None

    def execute_logic(self, symbol, pred_ret, bar_context, ts):
        state = self.states[symbol]
        current_price = bar_context['close']
        
        # Apply ShinkaEvolve Logic
        out = generate_signal(pred_ret, bar_context)
        signal = out['signal']
        
        action = "HOLD"
        pnl_to_log = 0.0
        fee_to_log = 0.0
        
        # 1. Check Exits
        exit_reason = None
        if state['position'] != 0:
            state['bars_held'] += 1
            pnl_pct = (current_price - state['entry_price']) / state['entry_price'] * state['position']
            
            if pnl_pct >= state['take_profit']:
                exit_reason = "TP"
            elif pnl_pct <= -state['stop_loss']:
                exit_reason = "SL"
            elif state['bars_held'] >= state['max_bars']:
                exit_reason = "TIME"
            elif signal != 0 and signal != state['position']:
                exit_reason = "FLIP"
            elif signal == 0:
                exit_reason = "SIGNAL"
        
        if exit_reason:
            notional_exit = state['qty'] * current_price
            exit_fee = notional_exit * FEE_RATE
            pnl = state['qty'] * (current_price - state['entry_price']) * state['position']
            net_pnl = pnl - exit_fee # Round-trip entry fee was deducted at entry
            
            state['equity'] += pnl - exit_fee
            state['realized_pnl'] += net_pnl
            pnl_to_log = net_pnl
            fee_to_log = exit_fee
            
            logger.info(f"[{symbol}] EXIT {exit_reason} at {current_price:.4f}. PnL: ${pnl:.4f}, Fee: ${exit_fee:.4f}, Net PnL: ${net_pnl:.4f}")
            state['position'] = 0
            state['qty'] = 0.0
            state['bars_held'] = 0
            action = f"EXIT_{exit_reason}"

        # 2. Open Position
        if signal != 0 and state['position'] == 0:
            state['position'] = signal
            state['entry_price'] = current_price
            state['bars_held'] = 0
            state['take_profit'] = out['take_profit']
            state['stop_loss'] = out['stop_loss']
            state['max_bars'] = out['max_bars']
            
            # Sizing: Identical to backtester.py
            notional = state['equity'] * out['position_size']
            # Cap by leverage (Original Backtester MAX_LEVERAGE = 3.0)
            max_notional = state['equity'] * 3.0
            notional = min(notional, max_notional)
            
            entry_fee = notional * FEE_RATE
            
            state['equity'] -= entry_fee
            state['qty'] = notional / current_price
            fee_to_log = entry_fee
            
            logger.info(f"[{symbol}] ENTER {'LONG' if signal == 1 else 'SHORT'} at {current_price:.4f}. Size: ${notional:.2f}, Fee: ${entry_fee:.4f}")
            action = "ENTER_LONG" if signal == 1 else "ENTER_SHORT"

        # Update Unrealized
        if state['position'] != 0:
            state['unrealized_pnl'] = state['qty'] * (current_price - state['entry_price']) * state['position']
        else:
            state['unrealized_pnl'] = 0.0

        # Log to CSV
        self._log_trade(symbol, ts, current_price, pred_ret, action, state['position'], fee_to_log, pnl_to_log)

    def _log_trade(self, symbol, ts, price, prediction, action, position, fee, pnl):
        safe_name = symbol.replace('/', '_')
        path = os.path.join(BASE_DIR, f'trades_{safe_name}.csv')
        df = pd.DataFrame([[ts, price, prediction, action, position, fee, pnl]], 
                          columns=['timestamp', 'price', 'prediction', 'action', 'position', 'fee', 'pnl'])
        df.to_csv(path, mode='a', header=False, index=False)

    def update_totals(self, ts):
        total_equity = sum(s['equity'] + s['unrealized_pnl'] for s in self.states.values())
        total_pnl = total_equity - INITIAL_TOTAL_EQUITY
        
        path = os.path.join(BASE_DIR, 'total_pnl.csv')
        df = pd.DataFrame([[ts, total_equity, total_pnl]], 
                          columns=['timestamp', 'total_equity', 'total_pnl'])
        df.to_csv(path, mode='a', header=False, index=False)
        
        self.total_equity = total_equity
        self.total_pnl = total_pnl
        
        logger.info(f"--- PORTFOLIO UPDATE ---")
        logger.info(f"Total Equity: ${total_equity:.2f} | Total PnL: ${total_pnl:.2f}")

    def run_step(self):
        now = datetime.utcnow()
        logger.info(f"Step triggered at {now} UTC")
        
        for symbol in SYMBOLS:
            df = self.fetch_data(symbol)
            if df is None: continue
            
            pred_ret, bar_context = self.predict(df)
            if pred_ret is None: continue
            
            self.execute_logic(symbol, pred_ret, bar_context, now)
            
        self.update_totals(now)

def wait_for_next_interval():
    """Wait until the next 00, 15, 30, or 45 minute mark + 10s safety delay."""
    SAFETY_DELAY = 10 
    now = datetime.utcnow()
    minutes = now.minute
    seconds = now.second
    
    wait_mins = 15 - (minutes % 15)
    
    if wait_mins == 15 and seconds < SAFETY_DELAY:
        wait_secs = SAFETY_DELAY - seconds
    elif wait_mins == 15 and seconds >= SAFETY_DELAY:
        return 
    else:
        target = now.replace(second=0, microsecond=0) + timedelta(minutes=wait_mins, seconds=SAFETY_DELAY)
        wait_secs = (target - now).total_seconds()
    
    logger.info(f"Waiting {wait_secs:.1f}s for candle finalization (Target: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC)...")
    time.sleep(wait_secs)

def main():
    trader = PaperTrader()
    logger.info("Starting Paper Trading Script (Kraken Futures Mode)...")
    
    while True:
        wait_for_next_interval()
        try:
            trader.run_step()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        time.sleep(30)

if __name__ == "__main__":
    main()

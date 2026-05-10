"""
evaluate.py — ShinkaEvolve harness for crypto trading strategy evolution.

ShinkaEvolve calls:
    python evaluate.py --program_path <program_path> --results_dir <results_dir>

The candidate program must expose generate_signal(predicted_return, bar_context).
"""
import sys
import os
import json
import importlib.util
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import argparse

# Add backtester_v1 to path so we can import backtester machinery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from shinka.core import run_shinka_eval

# ─── EVALUATION SETTINGS ─────────────────────────────────────────────────────
TICKERS     = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "multi")
FEE_RATE    = 0.0005   # 0.05% taker fee per side
INIT_EQUITY = 10_000.0

# Paths to immutable artifacts
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_regression_v1.json")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler_v1.joblib")
CALIB_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "calibration_v1.joblib")

# ─── LOCAL IMPLEMENTATION OF RUN_STRATEGY ───
def run_strategy(df: pd.DataFrame, signal_fn, init_equity: float, fee_rate: float) -> dict:
    # 1. Load artifacts
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    calibration_factor = joblib.load(CALIB_PATH)
    
    # 2. Build features and scale
    from backtester_v1.scripts.feature_engineering import FEATURE_NAMES
    X = df[FEATURE_NAMES]
    X = X.fillna(0)
    # We use unscaled features for bar_context to make them readable for the LLM/Strategy
    # But we need scaled features for the model prediction
    X_scaled = scaler.transform(X)
    
    # 3. Predict returns
    dmatrix = xgb.DMatrix(X_scaled, feature_names=FEATURE_NAMES)
    bps_raw = model.predict(dmatrix)
    bps_cal = bps_raw * calibration_factor
    predicted_returns = bps_cal / 10000.0
    
    # 4. Simulation Loop
    equity = init_equity
    position = 0
    entry_price = 0.0
    qty = 0.0
    equity_curve = []
    trades = []
    
    # Prepare all data as numpy arrays for speed
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values
    
    # Convert all features to a dict of arrays
    feature_arrays = {col: df[col].values for col in FEATURE_NAMES}
    
    for i in range(len(df)):
        price_now = prices[i]
        pred_ret = predicted_returns[i]
        
        # Expand bar_context with as many indicators as possible
        bar_context = {
            "open":   opens[i],
            "high":   highs[i],
            "low":    lows[i],
            "close":  price_now,
            "volume": volumes[i],
        }
        # Add all engineered features (unscaled for readability)
        for col in FEATURE_NAMES:
            bar_context[col] = feature_arrays[col][i]
        
        # Alias common names if they exist in FEATURE_NAMES
        if 'atr_14' in bar_context: bar_context['atr'] = bar_context['atr_14']
        if 'rsi_14' in bar_context: bar_context['rsi'] = bar_context['rsi_14']
        
        out = signal_fn(pred_ret, bar_context)
        sig = out.get("signal", 0)
        pos_size_frac = out.get("position_size", 0.0)
        
        if position != 0 and sig != position:
            pnl = qty * (price_now - entry_price) * position
            fees = (qty * entry_price + qty * price_now) * fee_rate
            net_pnl = pnl - fees
            equity += net_pnl
            trades.append({"pnl": net_pnl, "entry_price": entry_price, "exit_price": price_now, "direction": position})
            position = 0
            qty = 0.0
            
        if sig != 0 and position == 0:
            position = sig
            entry_price = price_now
            notional = equity * pos_size_frac
            qty = notional / price_now
            
        unrealized = 0
        if position != 0:
            unrealized = qty * (price_now - entry_price) * position
        equity_curve.append(equity + unrealized)
        
    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": {"num_trades": len(trades)}
    }


def run_backtest(ticker: str, generate_signal_fn) -> dict:
    from backtester_v1.scripts.feature_engineering import build_features
    data_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(data_path)
    # build_features returns scaled df if scaler passed, but we want unscaled for context
    df_feat = build_features(df, scaler=None)
    df_combined = df_feat.join(df[['open', 'high', 'low', 'close', 'volume']], how='inner')
    return run_strategy(df_combined, generate_signal_fn, INIT_EQUITY, FEE_RATE)


def validate_fn(run_output) -> tuple[bool, str | None]:
    if run_output is None: return False, "run_backtest returned None"
    metrics = run_output.get("metrics", {})
    if metrics.get("num_trades", 0) < 5: return False, "Too few trades"
    equity_curve = run_output.get("equity_curve", [])
    if len(equity_curve) == 0: return False, "Empty equity curve"
    if not np.isfinite(equity_curve[-1]) or equity_curve[-1] <= 0: return False, "Invalid final equity"
    return True, None


def _compute_sharpe(equity_curve: list, bars_per_year: int = 35040) -> float:
    returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)
    if returns.std() == 0: return 0.0
    return float(np.sqrt(bars_per_year) * returns.mean() / (returns.std() + 1e-9))


def _compute_max_drawdown(equity_curve: list) -> float:
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-9)
    return float(abs(dd.min()) * 100)


def aggregate_fn(results: list, results_dir: str) -> dict:
    per_ticker = {}
    for ticker, run_output in zip(TICKERS, results):
        eq, trd = run_output["equity_curve"], run_output["trades"]
        n_trades = len(trd)
        winning = [t for t in trd if t["pnl"] > 0]
        win_rate = len(winning) / n_trades if n_trades > 0 else 0.0
        total_ret = (eq[-1] / INIT_EQUITY - 1.0) * 100.0
        max_dd = _compute_max_drawdown(eq)
        sharpe = _compute_sharpe(eq)
        
        per_ticker[ticker] = {
            "sharpe_ratio": sharpe,
            "total_return_pct": round(total_ret, 3),
            "max_drawdown_pct": round(max_dd, 3),
            "win_rate": round(win_rate, 4),
            "num_trades": n_trades,
            "calmar_ratio": round((total_ret/100)/abs(max_dd/100 + 1e-9), 4),
            "profit_factor": round(sum(t["pnl"] for t in winning)/abs(sum(t["pnl"] for t in trd if t["pnl"] < 0) + 1e-9), 4),
            "avg_trade_pnl": round(float(np.mean([t["pnl"] for t in trd])), 5) if trd else 0.0
        }

    keys = ["sharpe_ratio","total_return_pct","max_drawdown_pct","win_rate","num_trades","calmar_ratio","profit_factor","avg_trade_pnl"]
    agg = {k: float(np.mean([per_ticker[t][k] for t in TICKERS])) for k in keys}
    agg["num_trades"] = int(round(agg["num_trades"]))

    fitness = round((0.5 * agg["sharpe_ratio"] + 0.3 * agg["calmar_ratio"] + 0.2 * agg["win_rate"]) * min(1.0, agg["num_trades"]/50.0) * max(0.0, 1.0 - max(0.0, agg["max_drawdown_pct"]-15.0)/30.0), 6)

    summary_path = os.path.join(results_dir, "per_ticker_metrics.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"fitness": fitness, "aggregate": agg, "per_ticker": per_ticker}, f, indent=2)

    return {
        "combined_score": fitness,
        "public": {
            "sharpe": round(agg["sharpe_ratio"], 4),
            "total_ret": round(agg["total_return_pct"], 2),
            "max_dd": round(agg["max_drawdown_pct"], 2),
            "win_rate":    round(agg["win_rate"], 4),
            "num_trades":  agg["num_trades"],
            "profit_factor": round(agg["profit_factor"], 3),
        },
        "private": {"per_ticker": per_ticker}
    }


def main(program_path: str, results_dir: str):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    generate_signal_fn = mod.generate_signal

    metrics, correct, error_msg = run_shinka_eval(
        program_path=os.path.abspath(__file__),
        results_dir=results_dir,
        experiment_fn_name="run_backtest",
        num_runs=len(TICKERS),
        run_workers=1,
        get_experiment_kwargs=lambda idx: {"ticker": TICKERS[idx], "generate_signal_fn": generate_signal_fn},
        validate_fn=validate_fn,
        aggregate_metrics_fn=lambda res: aggregate_fn(res, results_dir),
    )
    return metrics, correct, error_msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    
    m, ok, err = main(args.program_path, args.results_dir)
    print(json.dumps({"metrics": m, "correct": ok, "error": err}, indent=2))

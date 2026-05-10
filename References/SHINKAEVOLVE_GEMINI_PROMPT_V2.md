# Task: Build a Fully Automated ShinkaEvolve Trading Logic Evolver

You are a senior Python engineer. Your job is to wire up the `shinka-evolve`
framework (pip-installable, from SakanaAI) to evolve the trading logic of an
XGBoost-powered cryptocurrency backtester. The entire loop — proposal,
evaluation, scoring, archiving — must run **without any human interaction**.

Read every section before writing a single line of code.

Run everything in the crypto-venv virtual environment.

---

## 0. WHAT ALREADY EXISTS (DO NOT TOUCH)

The project root contains `backtester_v1/`:

```
backtester_v1/
├── models/
│   ├── base_strategy.py          # Abstract base — immutable
│   ├── xgb_strategy.py           # Active strategy — this is what we evolve
│   ├── xgb_regression_v1.json    # Trained XGBoost weights — immutable
│   ├── scaler_v1.joblib           # Feature scaler — immutable
│   └── calibration_v1.joblib     # Calibration — immutable
├── scripts/
│   ├── backtester.py             # Simulation engine — immutable
│   ├── feature_engineering.py   # Feature pipeline — immutable
│   └── report.py                # Metrics — immutable
└── data/raw/multi/
    ├── BTC_USDT.parquet          # 15-min OHLCV candles
    ├── ETH_USDT.parquet
    ├── SOL_USDT.parquet
    └── ...                       # 10 coins total
```

The XGBoost model predicts the **15-minute forward return** of a cryptocurrency
as a continuous float. The model is already trained. You must never retrain it,
never modify the feature pipeline, and never alter the backtester engine.

---

## 1. INSTALL SHINKAEVOLVE

```bash
pip install shinka-evolve
# Verify
python -c "from shinka.core import ShinkaEvolveRunner; print('shinka OK')"
```

Create a `.env` file at the project root with your API keys:

```
GOOGLE_API_KEY=your-gemini-key-here
OPENAI_API_KEY=sk-...             # optional fallback
```

---

## 2. CREATE THE TASK DIRECTORY

Create `backtester_v1/evolve_task/` with exactly these files:

```
backtester_v1/evolve_task/
├── initial.py      ← seed trading logic (YOU WRITE)
├── evaluate.py     ← shinka evaluation harness (YOU WRITE)
└── run_evo.py      ← ShinkaEvolveRunner launcher (YOU WRITE)
```

---

## 3. WRITE `initial.py` — THE SEED STRATEGY

This file defines the **starting point** for evolution. ShinkaEvolve will only
mutate code inside the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers.
Everything outside those markers is immutable scaffolding.

### Rules for `initial.py`
- The evolve block must contain one callable: `generate_signal(predicted_return, bar_context) -> dict`
- `predicted_return` is a float (the XGBoost model output, e.g. `0.0031` = +0.31% predicted)
- `bar_context` is a dict with keys: `close`, `high`, `low`, `volume`, `atr`, `rsi`, `adx`
- The function must return a dict with these exact keys:
  ```python
  {
      "signal":        int,    # 1=long, -1=short, 0=flat
      "position_size": float,  # fraction of portfolio [0.0, 1.0]
      "take_profit":   float,  # price distance, e.g. 0.004 = 0.4%
      "stop_loss":     float,  # price distance, e.g. 0.003 = 0.3%
      "max_bars":      int,    # time-based exit after N bars
  }
  ```
- The evolve block may import numpy as np and math. No other imports inside the block.
- Set `random.seed(42)` and `numpy.random.seed(42)` at the top of the file (outside the block).

### Seed logic to use as `initial.py`

```python
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
```

---

## 4. WRITE `evaluate.py` — THE SHINKA EVALUATION HARNESS

This is the fitness function. ShinkaEvolve calls it once per candidate program.

### Required structure

`evaluate.py` must follow the `run_shinka_eval` contract exactly:

```python
from shinka.core import run_shinka_eval

def main(program_path: str, results_dir: str):
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_backtest",   # function called inside the candidate
        num_runs=1,
        run_workers=1,
        get_experiment_kwargs=get_kwargs_fn,
        validate_fn=validate_fn,
        aggregate_metrics_fn=aggregate_fn,
    )
    return metrics, correct, error_msg
```

### Full `evaluate.py` implementation to write

```python
"""
evaluate.py — ShinkaEvolve harness for crypto trading strategy evolution.

ShinkaEvolve calls:
    python evaluate.py <program_path> <results_dir>

The candidate program must expose generate_signal(predicted_return, bar_context).
"""
import sys
import os
import json
import importlib.util
import numpy as np
import pandas as pd

# Add backtester_v1 to path so we can import backtester machinery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from shinka.core import run_shinka_eval

# ─── EVALUATION SETTINGS ─────────────────────────────────────────────────────
TICKERS     = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "multi")
FEE_RATE    = 0.0005   # 0.05% taker fee per side
INIT_EQUITY = 10_000.0


# ─── STEP 1: DEFINE WHAT ONE BACKTEST RUN DOES ───────────────────────────────
def run_backtest(ticker: str, generate_signal_fn) -> dict:
    """
    Runs a full backtest for one ticker using generate_signal_fn as the strategy.
    Returns a dict of per-trade and equity-level metrics.
    """
    from backtester_v1.scripts.feature_engineering import build_features
    from backtester_v1.scripts.backtester import run_strategy

    data_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(data_path)

    # Build features (model predictions included)
    df_feat = build_features(df)

    # Run the simulation engine — pass our signal function as the strategy hook
    result = run_strategy(
        df=df_feat,
        signal_fn=generate_signal_fn,
        init_equity=INIT_EQUITY,
        fee_rate=FEE_RATE,
    )

    # result is expected to be a dict: {equity_curve, trades, metrics}
    return result


# ─── STEP 2: KWARGS FACTORY (tells shinka what args to pass to run_backtest) ─
def get_kwargs_fn(program_path: str):
    """
    Loads generate_signal from the candidate program and returns one kwargs dict
    per ticker. shinka will call run_backtest(**kwargs) for each.
    """
    # Dynamically load the candidate module
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    generate_signal_fn = mod.generate_signal

    # Return a list of kwargs dicts — one per ticker
    return [
        {"ticker": ticker, "generate_signal_fn": generate_signal_fn}
        for ticker in TICKERS
    ]


# ─── STEP 3: VALIDATION ──────────────────────────────────────────────────────
def validate_fn(run_output) -> tuple[bool, str | None]:
    """
    Checks that the run_backtest output is structurally valid.
    Returns (is_valid, error_msg_or_None).
    """
    if run_output is None:
        return False, "run_backtest returned None"

    metrics = run_output.get("metrics", {})
    num_trades = metrics.get("num_trades", 0)

    if num_trades < 5:
        return False, f"Too few trades ({num_trades}); strategy may be degenerate"

    equity_curve = run_output.get("equity_curve", [])
    if len(equity_curve) == 0:
        return False, "Empty equity curve returned"

    final_equity = equity_curve[-1]
    if not np.isfinite(final_equity) or final_equity <= 0:
        return False, f"Non-finite or zero final equity: {final_equity}"

    return True, None


# ─── STEP 4: METRICS AGGREGATION ─────────────────────────────────────────────
def _compute_sharpe(equity_curve: list, bars_per_year: int = 35040) -> float:
    """Annualized Sharpe from equity curve (15-min bars → 35040/yr)."""
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(bars_per_year) * returns.mean() / returns.std())


def _compute_max_drawdown(equity_curve: list) -> float:
    """Peak-to-trough max drawdown as a positive percentage."""
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(abs(dd.min()) * 100)


def _compute_calmar(total_return_pct: float, max_dd_pct: float) -> float:
    if max_dd_pct == 0:
        return 0.0
    return float((total_return_pct / 100) / (max_dd_pct / 100))


def _compute_profit_factor(trades: list) -> float:
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss   = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def _compute_fitness(agg: dict) -> float:
    """
    Composite fitness scalar. Higher is better.
    Penalizes low trade count and excessive drawdown.
    """
    sharpe   = agg["sharpe_ratio"]
    calmar   = agg["calmar_ratio"]
    win_rate = agg["win_rate"]
    n_trades = agg["num_trades"]
    dd       = agg["max_drawdown_pct"]

    # Penalty: too few trades = insufficient signal
    trade_penalty = min(1.0, n_trades / 50.0)

    # Penalty: drawdown beyond 15% is dangerous
    dd_penalty = max(0.0, 1.0 - max(0.0, dd - 15.0) / 30.0)

    return round(
        (0.5 * sharpe + 0.3 * calmar + 0.2 * win_rate) * trade_penalty * dd_penalty,
        6,
    )


def aggregate_fn(results: list, results_dir: str) -> dict:
    """
    Aggregates results from all tickers into a single metrics dict.
    Must return: {"combined_score": float, "public": dict, "private": dict}
    combined_score is the PRIMARY fitness used by ShinkaEvolve.
    """
    per_ticker = {}
    for ticker, run_output in zip(TICKERS, results):
        eq  = run_output["equity_curve"]
        trd = run_output["trades"]
        m   = run_output["metrics"]

        n_trades    = len(trd)
        winning     = [t for t in trd if t["pnl"] > 0]
        win_rate    = len(winning) / n_trades if n_trades > 0 else 0.0
        total_ret   = (eq[-1] / INIT_EQUITY - 1.0) * 100.0
        max_dd      = _compute_max_drawdown(eq)
        sharpe      = _compute_sharpe(eq)
        calmar      = _compute_calmar(total_ret, max_dd)
        avg_pnl     = float(np.mean([t["pnl"] for t in trd])) if trd else 0.0
        pf          = _compute_profit_factor(trd)

        per_ticker[ticker] = {
            "sharpe_ratio":      sharpe,
            "total_return_pct":  round(total_ret, 3),
            "max_drawdown_pct":  round(max_dd, 3),
            "win_rate":          round(win_rate, 4),
            "num_trades":        n_trades,
            "avg_trade_pnl":     round(avg_pnl, 5),
            "profit_factor":     round(pf, 4),
            "calmar_ratio":      round(calmar, 4),
        }

    # Cross-ticker averages
    keys = ["sharpe_ratio","total_return_pct","max_drawdown_pct","win_rate",
            "num_trades","avg_trade_pnl","profit_factor","calmar_ratio"]
    agg = {k: float(np.mean([per_ticker[t][k] for t in TICKERS])) for k in keys}
    agg["num_trades"] = int(round(agg["num_trades"]))

    fitness = _compute_fitness(agg)

    # Save a human-readable breakdown alongside shinka's internal artifacts
    summary_path = os.path.join(results_dir, "per_ticker_metrics.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"fitness": fitness, "aggregate": agg, "per_ticker": per_ticker}, f, indent=2)

    return {
        "combined_score": fitness,           # ← shinka's fitness signal
        "public": {                          # ← visible in WebUI & logs
            "sharpe":      round(agg["sharpe_ratio"], 4),
            "total_ret":   round(agg["total_return_pct"], 2),
            "max_dd":      round(agg["max_drawdown_pct"], 2),
            "win_rate":    round(agg["win_rate"], 4),
            "num_trades":  agg["num_trades"],
            "profit_factor": round(agg["profit_factor"], 3),
        },
        "private": {                         # ← internal analysis only
            "calmar_ratio":  round(agg["calmar_ratio"], 4),
            "avg_trade_pnl": round(agg["avg_trade_pnl"], 6),
            "per_ticker":    per_ticker,
        },
    }


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
def main(program_path: str, results_dir: str):
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_backtest",
        num_runs=len(TICKERS),
        run_workers=1,
        get_experiment_kwargs=get_kwargs_fn,
        validate_fn=validate_fn,
        aggregate_metrics_fn=aggregate_fn,
    )
    return metrics, correct, error_msg


if __name__ == "__main__":
    prog_path   = sys.argv[1]
    results_dir = sys.argv[2]
    m, ok, err  = main(prog_path, results_dir)
    print(json.dumps({"metrics": m, "correct": ok, "error": err}, indent=2))
```

---

## 5. WRITE `run_evo.py` — THE AUTOMATED RUNNER

This is the **single command that starts the entire automated evolution**.
No human interaction after this is launched.

```python
"""
run_evo.py — Fully automated ShinkaEvolve launcher for crypto trading logic.

Usage:
    python backtester_v1/evolve_task/run_evo.py

ShinkaEvolve handles: proposal → eval → archive → next proposal, indefinitely
until num_generations or max_api_cost is reached.
"""
import os
import sys

# ── make sure backtester_v1 is on the path ───────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    # ── Job config: how to run evaluate.py for each candidate ────────────────
    job_config = LocalJobConfig(
        eval_program_path=os.path.join(HERE, "evaluate.py"),
        activate_script=os.path.join(HERE, "..", "..", ".venv", "bin", "activate"),
        # If using conda instead: conda_env="shinka"
    )

    # ── Database config: island-based archive ────────────────────────────────
    db_config = DatabaseConfig(
        archive_size=40,             # keep top-40 solutions in the archive
        num_archive_inspirations=2,  # use 2 archive solutions as LLM context
        num_islands=2,               # 2 parallel evolution islands
        migration_interval=10,       # cross-pollinate every 10 generations
    )

    # ── Evolution config: LLM mutation settings ───────────────────────────────
    evo_config = EvolutionConfig(
        init_program_path=os.path.join(HERE, "initial.py"),
        language="python",
        num_generations=100,
        # Use Gemini Flash as primary mutator; GPT-4o as fallback
        # ShinkaEvolve uses a UCB1 bandit to allocate calls between models
        llm_models=[
            "gemini/gemini-2.5-flash-preview-05-20",
            "gemini/gemini-2.5-pro-preview-05-06",
        ],
        task_sys_msg=(
            "You are an expert quantitative trader and Python engineer. "
            "You are evolving a crypto trading signal function that sits on top of "
            "a trained XGBoost model which predicts the 15-minute forward return of "
            "a cryptocurrency. The function receives (predicted_return: float, "
            "bar_context: dict) and must return a dict with keys: signal (1/0/-1), "
            "position_size (0.0-1.0), take_profit (float), stop_loss (float), "
            "max_bars (int). Fitness is a composite of Sharpe ratio, Calmar ratio, "
            "and win rate, penalized for too few trades or excessive drawdown. "
            "Focus mutations on: entry thresholds, position sizing logic, "
            "volatility-adjusted stops, trend filters using bar_context fields "
            "(atr, rsi, adx), signal smoothing, and asymmetric long/short rules. "
            "Do NOT import external libraries beyond numpy and math inside the "
            "EVOLVE-BLOCK. Do NOT look ahead — only use bar_context values available "
            "at bar close."
        ),
        results_dir=os.path.join(HERE, "results"),
    )

    # ── Runner: async proposal + eval concurrency ─────────────────────────────
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=3,   # 3 backtests running in parallel
        max_proposal_jobs=2,     # 2 LLM mutation calls in parallel
        max_db_workers=2,
    )

    print("=" * 60)
    print("ShinkaEvolve — Crypto Trading Logic Evolver")
    print(f"Task dir:    {HERE}")
    print(f"Results dir: {evo_config.results_dir}")
    print(f"Generations: {evo_config.num_generations}")
    print(f"Islands:     {db_config.num_islands}")
    print(f"Models:      {evo_config.llm_models}")
    print("=" * 60)

    runner.run()

    print("\nEvolution complete. Check results/ for the best strategy.")
    print("Launch WebUI with: shinka_webui --results-dir", evo_config.results_dir)


if __name__ == "__main__":
    main()
```

---

## 6. HOW TO RUN (FULLY AUTOMATED — ONE COMMAND)

After writing all three files:

```bash
# From the project root
cd /path/to/your/project

# Install shinka (once)
pip install shinka-evolve

# Set up your .env with API keys (once)
echo "GOOGLE_API_KEY=your-key-here" >> .env

# Verify the backtester still works with the seed strategy
python -c "
import sys; sys.path.insert(0, '.')
from backtester_v1.evolve_task.evaluate import main
m, ok, err = main('backtester_v1/evolve_task/initial.py', '/tmp/shinka_test')
print('Seed fitness:', m.get('combined_score'))
print('Valid:', ok)
print('Error:', err)
"

# Launch the full automated evolution (runs until 100 generations or Ctrl+C)
python backtester_v1/evolve_task/run_evo.py

# Monitor live progress in your browser (open a second terminal)
shinka_webui --results-dir backtester_v1/evolve_task/results
```

To resume after an interruption (shinka restores from checkpoint automatically):
```bash
python backtester_v1/evolve_task/run_evo.py
# results_dir points to the same path → shinka detects prior run and resumes
```

---

## 7. WHAT TO DELIVER

Write all three files exactly as specified:

1. `backtester_v1/evolve_task/initial.py` — seed strategy with EVOLVE-BLOCK markers
2. `backtester_v1/evolve_task/evaluate.py` — full shinka evaluation harness
3. `backtester_v1/evolve_task/run_evo.py` — ShinkaEvolveRunner launcher

Then verify:
- `python -c "from shinka.core import run_shinka_eval; print('OK')"` passes
- The seed evaluation (`main('initial.py', '/tmp/test')`) returns a valid non-None fitness
- `run_evo.py` runs without import errors

Do **not** modify anything in `backtester_v1/models/`, `backtester_v1/scripts/`,
or `backtester_v1/data/`. Those are immutable.

---

## 8. IMPORTANT CONSTRAINTS

| Rule | Detail |
|------|--------|
| No look-ahead | `bar_context` contains only values available at bar close `t` |
| No external libs in EVOLVE-BLOCK | Only `numpy` and `math` are allowed inside the mutable block |
| Fee must be applied | `FEE_RATE = 0.0005` (0.05% per side) — already set in `evaluate.py` |
| Evaluation set is fixed | Always the same 3 tickers and full parquet history — ensures comparable fitness |
| `combined_score` drives evolution | This is the only number ShinkaEvolve optimizes; make sure it is well-scaled |
| Seeds are reproducible | `random.seed(42)` and `np.random.seed(42)` at module level in `initial.py` |

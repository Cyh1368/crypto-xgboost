"""
run_evo.py — Fully automated ShinkaEvolve launcher for crypto trading logic.

Usage:
    python backtester_v1/evolve_task/run_evo.py

ShinkaEvolve handles: proposal → eval → archive → next proposal, indefinitely
until num_generations or max_api_cost is reached.
"""
import os
import sys

# ── make sure crypto-xgboost is on the path ───────────────────────────────────
# HERE is crypto-xgboost/backtester_v1/evolve_task
# We want to add crypto-xgboost to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    # ── Job config: how to run evaluate.py for each candidate ────────────────
    # crypto-venv is at /home/chengyou1368/crypto-venv
    job_config = LocalJobConfig(
        eval_program_path=os.path.join(HERE, "evaluate.py"),
        activate_script=os.path.abspath(os.path.join(HERE, "..", "..", "..", "crypto-venv", "bin", "activate")),
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
        # Use Claude 3.5 Sonnet, Gemini 2.0 Flash, and GPT-4o as mutators
        llm_models=[
            # "anthropic/claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            # "gemini/gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gpt-5.4-mini",
        ],
        task_sys_msg=(
            "You are an expert quantitative trader and Python engineer. "
            "You are evolving a crypto trading signal function that sits on top of "
            "a trained XGBoost model which predicts the 15-minute forward return of "
            "a cryptocurrency. The function receives (predicted_return: float, "
            "bar_context: dict) and must return a dict with keys: signal (1/0/-1), "
            "position_size (0.0-1.0), take_profit (float), stop_loss (float), "
            "max_bars (int). Fitness is a composite of Sharpe ratio, Calmar ratio, "
            "and win rate. "
            "The bar_context is extremely rich and contains: "
            "1. OHLCV data. "
            "2. Microstructure: Order Book Imbalance (obi_tau1, 3, 5, 10), spread_bps, "
            "depth_ratio_5/10, book_pressure_3, and kyle_lambda_est. "
            "3. Price Action: Log returns (ret_1, 3, 6, 12, 48), Volatility (vol_5, 20, 60), "
            "RSI (6, 14), MACD Signal, Bollinger % (bb_pct), ATR (atr_14), "
            "Momentum, Wick Ratios, Volume Ratios, VWAP Deviation (vwap_dev), "
            "Autocorrelation (autocorr_5), Skew, Kurtosis, and Trend Strength. "
            "4. Macro: Funding Rate and its 8-hour MA. "
            "5. Time: Sin/Cos encodings for hour/day, session flags (is_asia_session, "
            "is_us_session, is_weekend), and minutes_to_funding. "
            "Focus mutations on using these indicators for entry filters, volatility-adjusted "
            "stops, and asymmetric logic. Do NOT import external libraries beyond numpy and "
            "math inside the EVOLVE-BLOCK. Do NOT look ahead."
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

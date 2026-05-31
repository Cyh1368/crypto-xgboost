# evolve/continue_211813.py

import argparse

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


TASK_SYS_MSG = """
You are optimizing an XGBoost trading model for 15-minute BTC/ETH/SOL perp futures.

The goal is to maximize out-of-sample directional accuracy (DA) on unseen validation data.
DA = fraction of bars where sign(predicted BPS) == sign(actual BPS).
Random baseline is 0.50. Target: val_DA >= 0.52 consistently.

You may modify:
  1. XGB_PARAMS - hyperparameters (depth, estimators, regularization, learning rate, etc.)
  2. build_features() - which features to compute, lookback windows, transformations,
     regime indicators, feature interactions, normalization methods.
  3. CLIP_PERCENTILE and MIN_PRED_STD_RATIO constants.

Constraints:
  - build_features() must use ONLY past data (shift by at least 1 before use)
  - Do not modify run_experiment() or the return format
  - Do not hardcode split indices or peek at test/val targets
  - Keep total feature count <= 80 to avoid curse of dimensionality
  - Execution time per candidate must stay under 3 minutes

Common failure modes to avoid:
  - pred_std_ratio < 0.15: model is predicting the mean, so loosen regularization or add signal features
  - overfit_gap > 0.10: train DA >> val DA, so increase regularization, reduce depth, or use fewer estimators
  - val_DA < 0.45: model is inverted on validation, so add regime-conditioning, regime features, or shorten lookbacks
"""


def build_runner(results_dir: str, num_generations: int) -> ShinkaEvolveRunner:
    job_config = LocalJobConfig(
        eval_program_path="evolve/evaluate.py",
        python_executable="./crypto-venv/bin/python",
    )

    db_config = DatabaseConfig(
        num_islands=4,
        archive_size=50,
        elite_selection_ratio=0.3,
        num_archive_inspirations=5,
        migration_interval=10,
        parent_selection_strategy="power_law",
    )

    evo_config = EvolutionConfig(
        init_program_path="evolve/initial.py",
        results_dir=results_dir,
        num_generations=num_generations,
        llm_models=[
            "claude-haiku-4-5-20251001",
            "gemini-3-flash-preview",
            "gpt-5.4-mini",
        ],
        llm_kwargs={"temperatures": [0.0, 0.5, 1.0], "max_tokens": 4096},
        use_text_feedback=True,
        patch_types=["diff", "full"],
        patch_type_probs=[0.8, 0.2],
        task_sys_msg=TASK_SYS_MSG,
    )

    return ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the ShinkaEvolve run in results_20260511_211813."
    )
    parser.add_argument(
        "--results-dir",
        default="results_20260511_211813",
        help="Existing Shinka results directory to resume.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=100,
        help=(
            "Total generation target, not additional generations. "
            "Use a value above the existing max generation to continue farther."
        ),
    )
    args = parser.parse_args()

    runner = build_runner(
        results_dir=args.results_dir,
        num_generations=args.num_generations,
    )
    runner.run()


if __name__ == "__main__":
    main()

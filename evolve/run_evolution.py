# evolve/run_evolution.py

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(
    eval_program_path="evolve/evaluate.py",
    python_executable="./crypto-venv/bin/python",
)

db_config = DatabaseConfig(
    num_islands=4,               # 4 parallel evolutionary islands
    archive_size=75,             # keep top-75 programs per island
    elite_selection_ratio=0.3,
    num_archive_inspirations=5,
    migration_interval=8,
    parent_selection_strategy="power_law",
)

evo_config = EvolutionConfig(
    init_program_path="results_20260511_211813/best/main.py",
    num_generations=150,
    llm_models=[
            # "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
            # "gemini/gemini-2.5-flash",
            # "gemini-3-flash-preview",
            "gpt-5.4-mini",
        ],
    use_text_feedback=True,      # feed the penalty breakdown back to LLM
    meta_rec_interval=20,
    meta_llm_models=["claude-sonnet-4-5-20250929"],
    llm_kwargs={"temperatures": [0.0, 0.5, 1.0], "max_tokens": 4096},
    meta_llm_kwargs={"temperatures": [0.0], "max_tokens": 4096},
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.8, 0.2],
    task_sys_msg="""
You are improving an XGBoost model for 15-minute BTC/ETH/SOL perp futures.

CURRENT BEST (do not regress below this):
  train_DA=0.572  test_DA=0.525  val_DA=0.511
  train_r=0.476   test_r=0.108   val_r=0.054
  val_slope=0.006  (near-zero — primary problem)
  pred_std_ratio_val=0.112

PRIMARY PROBLEM TO SOLVE:
The model's predictive slope collapses from 0.074 on train to 0.006 on val.
This means the model learns patterns that are specific to the training market
regime but evaporate in the most recent period. The test→val degradation is
sharper than the train→test degradation, suggesting the val period represents
a second regime shift.

STRATEGIES TO EXPLORE:
1. Shorter lookback windows (features based on 5-20 bars rather than 60+)
   decay faster but generalize better across regimes.
2. Z-score normalization of features within rolling windows makes them
   regime-invariant (a 1% move means different things in low vs high vol).
3. Features that measure the RATE OF CHANGE of market character
   (vol_trend, autocorrelation trend, efficiency ratio trend) rather than
   absolute levels.
4. Reducing n_estimators and increasing min_child_weight forces the model
   to learn only the most robust, high-sample patterns.
5. Adding a feature interaction: momentum × volatility_regime as a
   single multiplicative term often generalizes better than both separately.

TARGET METRICS:
  val_DA >= 0.525     (currently 0.511)
  val_slope >= 0.020  (currently 0.006)
  val_r >= 0.080      (currently 0.054)
  pred_std_ratio_val >= 0.18  (currently 0.112)
  train_DA - val_DA <= 0.06   (currently 0.061 — right at limit)

You may modify:
  1. XGB_PARAMS — hyperparameters (depth, estimators, regularization, learning rate, etc.)
  2. build_features() — which features to compute, lookback windows, transformations,
     regime indicators, feature interactions, normalization methods.
  3. CLIP_PERCENTILE and MIN_PRED_STD_RATIO constants.

Constraints:
  - build_features() must use ONLY past data
  - Do not modify run_experiment() or the return format
  - Do not hardcode split indices or peek at test/val targets
  - Do NOT use any feature that requires lookahead
  - Do NOT increase feature count above 80
  - Execution time per candidate must stay under 3 minutes
""",
)

if __name__ == "__main__":
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=4,
    )
    runner.run()

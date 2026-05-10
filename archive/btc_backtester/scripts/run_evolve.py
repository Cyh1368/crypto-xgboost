import argparse
import structlog
from btc_backtester.evolve.shinka_loop import ShinkaOrchestrator
from btc_backtester.evolve.claude_oracle import ClaudeOracle
from btc_backtester.data.loader import DataLoader
from btc_backtester.models.xgb_strategy import walk_forward_cv
from btc_backtester.backtester.engine import BacktestEngine
from btc_backtester.backtester.metrics import calculate_metrics

logger = structlog.get_logger()

MAX_ITERATIONS = 50
STAGNATION_LIMIT = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="btc_backtester/data/raw/btc_15m.parquet")
    parser.add_argument("--ledger", default="ledger.json")
    parser.add_argument("--dry-run", action="store_true", help="Generate mutation but don't apply")
    args = parser.parse_args()

    orchestrator = ShinkaOrchestrator(args.ledger)
    oracle = ClaudeOracle()
    loader = DataLoader(args.data)
    
    champion = orchestrator.ledger["champion"]
    stagnation = 0

    for i in range(MAX_ITERATIONS):
        print(f"\n═══ ShinkáEvolve Iteration {i+1} ═══")
        
        context = orchestrator.build_context_packet(champion, version=i)
        mutation = oracle.query_claude(context)
        
        print(f"  Mutation: [{mutation['type']}] {mutation['hypothesis']}")
        
        if args.dry_run:
            print("  [DRY-RUN] Skipping application and evaluation.")
            continue

        try:
            orchestrator.apply_patch(mutation["patch"])
        except Exception as e:
            print(f"  ❌ FAILED to apply patch: {e}")
            continue

        # Re-run backtest
        df = loader.load_data()
        results_df = walk_forward_cv(df)
        results_df['spread_bps'] = 2.0
        engine = BacktestEngine()
        final_df = engine.run(results_df)
        challenger = calculate_metrics(final_df)
        challenger["strategy_version"] = f"v{i+1}"

        if orchestrator.accept(champion, challenger):
            champion = challenger
            orchestrator.update_champion(challenger, mutation)
            print(f"  ✅ ACCEPTED  Sharpe {challenger['sharpe_annual']:.3f}")
            stagnation = 0
        else:
            orchestrator.revert_patch()
            stagnation += 1
            print(f"  ❌ REJECTED  Sharpe {challenger['sharpe_annual']:.3f} (stagnation={stagnation})")

        if stagnation >= STAGNATION_LIMIT:
            print("Stagnation limit reached. Halting evolution.")
            break

if __name__ == "__main__":
    main()

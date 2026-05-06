# BTC Futures Backtester

Production-grade 15-minute Bitcoin futures backtester with XGBoost-driven alpha signal and ShinkáEvolve loop.

## Features

- **XGBoost Strategy**: Advanced binary classification model for direction prediction.
- **SHAP-based Pruning**: Automatically drops features with low predictive power.
- **Microstructure Features**: Exponentially-decayed orderbook imbalance (OBI), Kyle's Lambda proxy, and more.
- **ShinkáEvolve**: Self-improving loop that uses Claude Code as an oracle to propose and evaluate strategy mutations.
- **Vectorized Backtester**: High-performance simulation with slippage and funding models.

## Repository Structure

```
btc_backtester/
├── data/                  # OHLCV + L2 snapshots
├── features/              # Feature implementation and registry
├── models/                # XGBoost and strategy logic
├── backtester/            # Engine, portfolio, and metrics
├── evolve/                # ShinkáEvolve orchestrator and oracle
├── scripts/               # Entrypoints
└── tests/                 # Unit tests
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r btc_backtester/requirements.txt
   ```

2. Configure environment variables in `.env`.

## Usage

### Initial Backtest
Runs the walk-forward cross-validation for the V0 strategy.
```bash
python btc_backtester/scripts/run_initial.py
```

### Evolution Loop
Starts the ShinkáEvolve process to iteratively improve the strategy.
```bash
python btc_backtester/scripts/run_evolve.py
```

### Dry Run
Generate mutations without applying them.
```bash
python btc_backtester/scripts/run_evolve.py --dry-run
```

## Success Criteria

- Sharpe (annual, OOS): > 1.0 (V0) / > 2.0 (Evolved)
- Max Drawdown: < 20% (V0) / < 12% (Evolved)

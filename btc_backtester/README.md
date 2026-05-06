# BTC Futures Backtester

Production-grade 15-minute Bitcoin futures backtester with XGBoost-driven alpha signal and ShinkáEvolve loop.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`.

## Usage

- Initial backtest: `python scripts/run_initial.py`
- Evolve loop: `python scripts/run_evolve.py`

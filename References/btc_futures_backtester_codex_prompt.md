# BTC Futures Backtester — Codex Development Prompt

> **Audience**: OpenAI Codex / Claude Code  
> **Mode**: Autonomous agentic development  
> **Evolve Loop**: ShinkáEvolve (described in §4)

---

## §0 — Mission

Build a **production-grade 15-minute Bitcoin futures backtester** with an XGBoost-driven alpha signal, exponentially-decayed orderbook imbalance features, and a self-improving **ShinkáEvolve** loop that uses Claude Code as an external oracle to propose, evaluate, and commit progressively better strategies.

---

## §1 — Repository Scaffold

Generate the following directory structure:

```
btc_backtester/
├── data/
│   ├── raw/                  # OHLCV + L2 orderbook snapshots (15-min)
│   └── processed/            # Feature matrices, labels
├── features/
│   ├── __init__.py
│   ├── orderbook.py          # Exponential-decay imbalance (§2.2)
│   ├── price_action.py       # Momentum, volatility, microstructure
│   ├── macro.py              # Funding rate, OI, basis
│   └── registry.py           # Central feature registry
├── models/
│   ├── xgb_strategy.py       # XGBoost wrapper + pruning (§2.3)
│   └── base_strategy.py      # ABC Strategy interface
├── backtester/
│   ├── engine.py             # Event-driven simulation
│   ├── portfolio.py          # Position sizing, margin tracking
│   └── metrics.py            # Sharpe, Calmar, max-DD, hit-rate
├── evolve/
│   ├── shinka_loop.py        # ShinkáEvolve orchestrator (§4)
│   ├── claude_oracle.py      # Claude Code subprocess caller
│   └── ledger.json           # Persistent strategy ledger
├── scripts/
│   ├── run_initial.py        # §3 entrypoint
│   └── run_evolve.py         # §4 entrypoint
├── tests/
│   └── test_features.py
├── requirements.txt
└── README.md
```

---

## §2 — Initial Strategy (V0)

### 2.1 Data Contract

- **Instrument**: BTC-PERP or BTC quarterly futures (Binance / Bybit / Deribit)
- **Bar size**: 15-minute OHLCV
- **Orderbook**: Top-20-level L2 snapshots aligned to bar close
- **Label**: `y = sign(close_{t+1} - close_t)` → binary classification (-1 / +1)
- **Train window**: rolling 90-day, re-fit every 7 days
- **Test window**: out-of-sample, walk-forward

---

### 2.2 Orderbook Imbalance Feature (mandatory)

Implement the exponentially-decayed weighted imbalance:

```
          Σ_{k=1}^{K}  bid_vol_k · e^{-k/τ}
OBI(τ) = ─────────────────────────────────────────────
          Σ_{k=1}^{K} (bid_vol_k + ask_vol_k) · e^{-k/τ}
```

**Requirements**:
- Compute for **τ ∈ {1, 3, 5, 10}** (four separate features)
- Normalize volumes by the rolling 1-hour average total volume
- Handle missing levels gracefully (zero-pad)
- Unit-test in `tests/test_features.py` with a synthetic order book

```python
# features/orderbook.py skeleton

import numpy as np

TAU_VALUES = [1, 3, 5, 10]

def obi(bids: np.ndarray, asks: np.ndarray, tau: float) -> float:
    """
    bids: shape (K, 2) — [[price, vol], ...]  sorted best→worst
    asks: shape (K, 2) — [[price, vol], ...]  sorted best→worst
    Returns OBI ∈ [0, 1].
    """
    K = min(len(bids), len(asks))
    levels = np.arange(1, K + 1)
    weights = np.exp(-levels / tau)
    bid_w = (bids[:K, 1] * weights).sum()
    ask_w = (asks[:K, 1] * weights).sum()
    denom = bid_w + ask_w
    return bid_w / denom if denom > 0 else 0.5


def all_obi_features(bids, asks) -> dict:
    return {f"obi_tau{tau}": obi(bids, asks, tau) for tau in TAU_VALUES}
```

---

### 2.3 XGBoost Feature Spam — Full Kitchen Sink

Implement **all** features below. Prune later; generate everything first.

#### 2.3.1 Price-Action Features (`features/price_action.py`)

| Feature | Description |
|---|---|
| `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_48` | Log-returns over N bars |
| `vol_5`, `vol_20`, `vol_60` | Rolling std of log-returns |
| `rsi_14`, `rsi_6` | Wilder RSI |
| `macd_signal` | MACD(12,26) minus signal(9) |
| `bb_pct` | Price position within Bollinger Bands(20,2) |
| `atr_14` | Average True Range normalized by close |
| `momentum_bar` | (close - open) / (high - low + ε) |
| `wick_ratio_up` | (high - max(open,close)) / atr |
| `wick_ratio_down` | (min(open,close) - low) / atr |
| `volume_ratio_5` | volume / rolling_mean(volume, 5) |
| `volume_ratio_20` | volume / rolling_mean(volume, 20) |
| `vwap_dev` | (close - vwap_20) / close |
| `autocorr_5` | 5-lag autocorrelation of returns |
| `skew_20` | 20-bar rolling skewness of returns |
| `kurt_20` | 20-bar rolling kurtosis of returns |
| `realized_vol_ratio` | vol_5 / vol_60 (vol-of-vol regime) |
| `trend_strength` | ADX(14) |
| `close_rank_48` | Percentile rank of close in last 48 bars |
| `gap_open` | (open_t - close_{t-1}) / close_{t-1} |
| `overnight_ret` | return during exchange low-volume window |

#### 2.3.2 Macro / Market-Structure Features (`features/macro.py`)

| Feature | Description |
|---|---|
| `funding_rate` | 8-hour funding rate at bar time |
| `funding_8h_ma` | 8-bar MA of funding rate |
| `oi_chg_1`, `oi_chg_6` | % change in open interest |
| `oi_vol_ratio` | OI / 24h volume |
| `basis_pct` | (futures_price - spot_price) / spot_price |
| `liquidation_buy_1h` | Buy-side liquidations in past 4 bars |
| `liquidation_sell_1h` | Sell-side liquidations in past 4 bars |
| `liq_imbalance` | (buy_liq - sell_liq) / (buy_liq + sell_liq + ε) |
| `fear_greed_idx` | Daily fear & greed index (forward-filled) |
| `btc_dominance_chg` | 1-day change in BTC dominance |

#### 2.3.3 Orderbook Microstructure (`features/orderbook.py`)

| Feature | Description |
|---|---|
| `obi_tau1/3/5/10` | See §2.2 |
| `spread_bps` | (ask1 - bid1) / mid × 10000 |
| `depth_ratio_5` | sum(bid_vol, top 5) / sum(ask_vol, top 5) |
| `depth_ratio_10` | same, top 10 |
| `mid_price_move` | (mid_t - mid_{t-1}) / mid_{t-1} |
| `book_pressure_3` | bid_vol[1:4].sum() / ask_vol[1:4].sum() |
| `vol_at_spread` | bid_vol[0] + ask_vol[0] normalized |
| `kyle_lambda_est` | ΔP / ΔV proxy (signed flow estimation) |

#### 2.3.4 Time / Calendar Features

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclical hour encoding |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding |
| `is_asia_session` | 00:00–08:00 UTC binary |
| `is_us_session` | 13:30–20:00 UTC binary |
| `is_weekend` | Binary |
| `minutes_to_funding` | Minutes until next 8h funding timestamp |

---

### 2.4 XGBoost Model Config

```python
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "tree_method": "hist",
    "device": "cuda",          # fall back to "cpu" if unavailable
    "random_state": 42,
}
```

**Feature pruning after first fit**:
1. Compute SHAP values on validation set
2. Drop features with `mean(|SHAP|) < 1e-4`
3. Re-train with pruned set; assert Sharpe does not drop > 0.05
4. Save pruned feature list to `features/pruned_v{version}.json`

---

### 2.5 Signal → Position Logic

```python
# Simple threshold signal
prob = model.predict_proba(X)[:, 1]   # P(up)

LONG_THRESH  = 0.58
SHORT_THRESH = 0.42
SIZE_SCALE   = 0.10   # 10% of NAV per trade, scale by |prob - 0.5|

signal = np.where(prob > LONG_THRESH, 1,
         np.where(prob < SHORT_THRESH, -1, 0))

position_size = signal * SIZE_SCALE * 2 * (np.abs(prob - 0.5))
```

---

### 2.6 Risk Controls (implement in `backtester/portfolio.py`)

- Max single position: 20% NAV
- Max drawdown kill-switch: halt trading if rolling 5-day DD > 8%
- Per-trade stop-loss: 1.5 × ATR(14)
- Take-profit: 2.5 × ATR(14)
- Slippage model: `0.5 × spread_bps + 0.5 bps` taker fee
- Funding cost: accrued per bar proportionally

---

## §3 — Backtest Engine

Implement a **vectorized-first, event-driven fallback** engine in `backtester/engine.py`.

**Required output metrics** (stored in `ledger.json` after each run):

```json
{
  "strategy_version": "v0",
  "sharpe_annual": 1.42,
  "calmar_ratio": 0.88,
  "max_drawdown_pct": -14.2,
  "win_rate": 0.524,
  "avg_trade_pct": 0.18,
  "trades_total": 1240,
  "feature_set": ["obi_tau3", "ret_6", "..."],
  "params": { "...": "..." },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

Run the initial backtest:

```bash
python scripts/run_initial.py \
  --data data/raw/btc_15m.parquet \
  --start 2022-01-01 \
  --end   2024-12-31 \
  --out   ledger.json
```

---

## §4 — ShinkáEvolve Loop

> **Shinkái** (深化) — Japanese for "deepening." The loop deepens understanding of what makes strategies work by iteratively proposing mutations, evaluating them, and keeping only improvements.

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ShinkáEvolve Orchestrator              │
│  evolve/shinka_loop.py                                   │
│                                                          │
│  1. Load ledger.json → current champion metrics          │
│  2. Build CONTEXT PACKET → send to Claude Code oracle    │
│  3. Claude Code proposes MUTATIONS (Python patch)        │
│  4. Apply patch → re-run backtester                      │
│  5. Compare metrics → accept if Sharpe improves ≥ +0.05  │
│  6. Update ledger, tag git commit, loop                  │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Context Packet (sent to Claude Code each iteration)

```python
CONTEXT_TEMPLATE = """
You are a quantitative researcher improving a BTC futures trading strategy.

## Current Champion (v{version})
- Sharpe (annual): {sharpe}
- Calmar: {calmar}
- Max DD: {max_dd}%
- Win rate: {win_rate}%
- Feature count: {n_features}
- Top-10 SHAP features: {top_shap}
- Bottom-5 SHAP features (candidates for removal): {bottom_shap}

## Strategy Code Snapshot
```python
{strategy_code_snippet}
```

## Backtest Config
{backtest_config_json}

## SHAP Summary (last fold)
{shap_summary_table}

## Task
Propose exactly ONE of the following mutations as a unified git diff / Python patch:

MUTATION TYPES (choose the most promising):
  A) ADD_FEATURE       — implement a new feature in features/
  B) REMOVE_FEATURE    — drop a low-SHAP feature
  C) TRANSFORM_FEATURE — apply transformation (log, rank, lag, cross)
  D) SIGNAL_THRESHOLD  — tune LONG_THRESH / SHORT_THRESH
  E) POSITION_SIZING   — modify SIZE_SCALE or Kelly fraction logic
  F) STOP_LOGIC        — modify ATR multipliers or trailing stop
  G) MODEL_PARAM       — change an XGBoost hyperparameter
  H) REGIME_FILTER     — add a market-regime gate (e.g., vol-of-vol)
  I) ENSEMBLE          — combine with a second model or signal

Rules:
- Output ONLY valid Python code / diff. No prose preamble.
- Mutations must be minimal and isolated.
- If adding a feature, include a docstring with its hypothesis.
- Do not change the data pipeline contract.

Respond with:
<MUTATION_TYPE>TYPE_LETTER</MUTATION_TYPE>
<HYPOTHESIS>One sentence on why this should improve Sharpe.</HYPOTHESIS>
<PATCH>
... python patch or new file content ...
</PATCH>
"""
```

### 4.3 Claude Code Oracle (`evolve/claude_oracle.py`)

```python
import subprocess, json, re

def query_claude(context: str, model: str = "claude-opus-4-5") -> dict:
    """
    Calls Claude Code via CLI subprocess.
    Returns parsed mutation dict: {type, hypothesis, patch}.
    """
    result = subprocess.run(
        ["claude", "-p", context, "--output-format", "text"],
        capture_output=True, text=True, timeout=120
    )
    raw = result.stdout

    mutation_type = re.search(r"<MUTATION_TYPE>(.*?)</MUTATION_TYPE>", raw, re.S)
    hypothesis    = re.search(r"<HYPOTHESIS>(.*?)</HYPOTHESIS>",       raw, re.S)
    patch         = re.search(r"<PATCH>(.*?)</PATCH>",                 raw, re.S)

    return {
        "type":       mutation_type.group(1).strip() if mutation_type else "UNKNOWN",
        "hypothesis": hypothesis.group(1).strip()    if hypothesis    else "",
        "patch":      patch.group(1).strip()         if patch         else "",
    }
```

### 4.4 Acceptance Criterion & Ledger Update

```python
ACCEPTANCE_RULES = {
    "min_sharpe_delta":   +0.05,   # must improve Sharpe by at least this
    "max_dd_regression":  -2.0,    # max drawdown must not worsen by more than 2%
    "min_trades":          200,    # reject degenerate strategies with few trades
}

def accept(champion: dict, challenger: dict) -> bool:
    return (
        challenger["sharpe_annual"]  >= champion["sharpe_annual"]  + ACCEPTANCE_RULES["min_sharpe_delta"]
        and challenger["max_drawdown_pct"] >= champion["max_drawdown_pct"] + ACCEPTANCE_RULES["max_dd_regression"]
        and challenger["trades_total"]     >= ACCEPTANCE_RULES["min_trades"]
    )
```

### 4.5 Evolution Loop Entrypoint

```python
# scripts/run_evolve.py

MAX_ITERATIONS = 50
STAGNATION_LIMIT = 10   # halt if no improvement for N consecutive iterations

def main():
    ledger = load_ledger("ledger.json")
    champion = ledger["champion"]
    stagnation = 0

    for i in range(MAX_ITERATIONS):
        print(f"\n═══ ShinkáEvolve Iteration {i+1} ═══")

        context  = build_context_packet(champion, version=i)
        mutation = query_claude(context)

        print(f"  Mutation: [{mutation['type']}] {mutation['hypothesis']}")

        apply_patch(mutation["patch"])
        challenger = run_backtest()

        if accept(champion, challenger):
            champion = challenger
            champion["strategy_version"] = f"v{i+1}"
            save_ledger(ledger, champion)
            git_commit(f"[ShinkáEvolve v{i+1}] {mutation['type']}: {mutation['hypothesis']}")
            print(f"  ✅ ACCEPTED  Sharpe {challenger['sharpe_annual']:.3f}")
            stagnation = 0
        else:
            revert_patch()
            stagnation += 1
            print(f"  ❌ REJECTED  Sharpe {challenger['sharpe_annual']:.3f} (stagnation={stagnation})")

        if stagnation >= STAGNATION_LIMIT:
            print("Stagnation limit reached. Halting evolution.")
            break

    print(f"\nFinal champion: {champion['strategy_version']} | Sharpe {champion['sharpe_annual']:.3f}")
```

---

## §5 — Implementation Instructions for Codex

Execute the following steps **in order**. Check off each before proceeding.

### Phase 1 — Scaffold & Data
- [ ] Generate repository structure from §1
- [ ] Implement data loader for Parquet OHLCV + L2 orderbook
- [ ] Write `features/registry.py` with lazy feature computation

### Phase 2 — Features
- [ ] Implement all features in §2.3 with type hints and docstrings
- [ ] Write unit tests covering edge cases (missing levels, zero volume, NaN propagation)
- [ ] Verify `obi_tau` values are bounded [0,1]

### Phase 3 — Model
- [ ] Implement `models/xgb_strategy.py` with SHAP-based pruning
- [ ] Implement rolling walk-forward cross-validation
- [ ] Save pruned feature list after each fold

### Phase 4 — Backtester
- [ ] Implement vectorized engine with slippage + funding model
- [ ] Implement metrics in `backtester/metrics.py`
- [ ] Output ledger JSON after each run

### Phase 5 — ShinkáEvolve
- [ ] Implement `evolve/claude_oracle.py` with retry/timeout
- [ ] Implement patch apply/revert using `git apply` + `git stash`
- [ ] Implement ledger update + git tagging
- [ ] Wire `scripts/run_evolve.py` end-to-end

### Phase 6 — Hardening
- [ ] Add logging (structlog or loguru) to all modules
- [ ] Add `--dry-run` flag to evolve script (generate mutation but don't apply)
- [ ] Add Telegram / Discord webhook notification on champion update
- [ ] Write `README.md` with setup and usage instructions

---

## §6 — Requirements (`requirements.txt`)

```
xgboost>=2.0
shap>=0.44
pandas>=2.1
numpy>=1.26
pyarrow>=14.0
scikit-learn>=1.4
optuna>=3.5          # optional: for HPO inside evolve loop
structlog>=24.0
python-dotenv>=1.0
requests>=2.31
anthropic>=0.25      # Claude SDK (alternative to CLI subprocess)
pytest>=8.0
```

---

## §7 — Success Criteria

| Metric | V0 Target | Evolved Target |
|---|---|---|
| Sharpe (annual, OOS) | > 1.0 | > 2.0 |
| Calmar ratio | > 0.7 | > 1.2 |
| Max drawdown | < 20% | < 12% |
| Win rate | > 50% | > 53% |
| Avg trade | > 0.10% | > 0.20% |
| Features (pruned) | ≤ 30 | ≤ 20 |

---

## §8 — Notes & Caveats for the Evolving Agent

1. **Never look ahead.** All features must be computable at bar-close `t` using only data available up to and including `t`. Enforce this with a `_check_lookahead()` assertion in the engine.

2. **Funding rate alignment.** Funding is paid at 00:00, 08:00, 16:00 UTC. Accrue proportionally per 15-min bar; do not lump at payment time.

3. **Survivorship.** If using aggregated exchange data, note that exchange delistings and contract rolls create artificial gaps. Handle rolls explicitly.

4. **Regime awareness.** Tag each bar with a volatility regime (low/mid/high, based on rolling realized vol percentile) and report per-regime Sharpe in the ledger.

5. **Overfitting guard.** If OOS Sharpe < 0.5 × IS Sharpe, flag the strategy as overfit and reject regardless of OOS absolute level.

6. **The oracle is not infallible.** Claude Code may propose syntactically valid but economically nonsensical mutations. The acceptance criterion in §4.4 is the final arbiter.

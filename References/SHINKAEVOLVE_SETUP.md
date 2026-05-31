# ShinkaEvolve Setup Guide — XGBoost Model Evolution

## Objective

Use ShinkaEvolve to automatically discover the best combination of:
- XGBoost hyperparameters
- Feature engineering logic (which features to include, lookback windows, transformations)
- Target preprocessing (clipping thresholds, BPS scaling)
- Regime-conditional logic

The evolutionary loop will mutate `initial.py` across generations, score each candidate
with `evaluate.py`, and converge toward programs that maximize out-of-sample directional
accuracy without overfitting or collapsing to mean prediction.

---

## Step 1 — Install ShinkaEvolve

```bash
git clone https://github.com/SakanaAI/ShinkaEvolve
cd ShinkaEvolve
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Also install project dependencies into this venv
uv pip install xgboost scikit-learn pandas numpy joblib
```

---

## Step 2 — File Structure

Create the following directory inside the project root (alongside `backtester_v2/`):

```
evolve/
├── evaluate.py          # Scorer — reads candidate, trains, returns metrics
├── initial.py           # Seed program — the LLM mutates the EVOLVE blocks
├── run_evolution.py     # Launcher script
└── results/             # Auto-created by ShinkaEvolve
```

Data paths used inside evaluate.py (already available):
```
backtester_v2/data/raw/BTC_USDT_real.parquet   (or multi-ticker equivalent)
backtester_v2/models/                           (write evolved models here)
```

---

## Step 3 — `initial.py` (Seed Program)

This file contains the seed logic that ShinkaEvolve will mutate. Mark every
section you want the LLM to modify with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`.
Do NOT mark data loading or the `run_experiment` interface — only the parts that
represent modeling choices.

```python
# evolve/initial.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (hyperparameters)
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

CLIP_PERCENTILE = 99          # clip targets at this percentile (both tails)
MIN_PRED_STD_RATIO = 0.15     # minimum acceptable pred_std / actual_std
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV + orderbook + funding data.
    All features must use only past data (no lookahead).
    Returns a DataFrame aligned to df.index.
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    # --- Returns ---
    for lag in [1, 3, 6, 12, 48]:
        feat[f'ret_{lag}'] = close.pct_change(lag)

    # --- Volatility ---
    log_ret = np.log(close / close.shift(1))
    for w in [5, 20, 60]:
        feat[f'vol_{w}'] = log_ret.rolling(w).std()

    # --- RSI ---
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_pct'] = (close - bb_mid) / (2 * bb_std + 1e-9)

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_norm'] = feat['atr_14'] / close

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feat['macd_signal'] = macd - macd.ewm(span=9, adjust=False).mean()

    # --- Volume ---
    feat['volume_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-9)
    feat['volume_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-9)
    vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat['vwap_dev'] = (close - vwap) / (close + 1e-9)

    # --- Session / Time ---
    feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    # --- Regime features ---
    feat['trend_strength_60'] = (close / close.shift(60) - 1).abs() * 10000
    realized_vol = log_ret.rolling(20).std()
    feat['vol_regime'] = realized_vol / (realized_vol.rolling(200).mean() + 1e-9)

    # --- Orderbook (if real data available) ---
    if 'bids' in df.columns and 'asks' in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.array(row['bids'])
                asks = np.array(row['asks'])
                bid_vol = np.sum(bids[:, 1] * np.exp(-tau * np.arange(len(bids))))
                ask_vol = np.sum(asks[:, 1] * np.exp(-tau * np.arange(len(asks))))
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            except Exception:
                return 0.0
        feat['obi_tau1'] = df.apply(lambda r: obi(r, 1), axis=1)
        feat['obi_tau3'] = df.apply(lambda r: obi(r, 3), axis=1)

    # --- Funding rate ---
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        feat['funding_rate'] = fr
        feat['funding_8h_ma'] = fr.rolling(32).mean()   # 32 × 15min = 8h

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
    return feat
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# DO NOT MODIFY BELOW THIS LINE
# run_experiment is the fixed interface called by evaluate.py

def run_experiment(
    X_train, y_train,
    X_test,  y_test,
    X_val,   y_val,
):
    """
    Train an XGBoost model and return raw predictions for all three splits.
    Called by evaluate.py via shinka.core.run_shinka_eval.
    """
    # Clip targets
    clip_val = np.percentile(np.abs(y_train), CLIP_PERCENTILE)
    y_train_c = y_train.clip(-clip_val, clip_val)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_va = scaler.transform(X_val)

    params = {k: v for k, v in XGB_PARAMS.items()
              if k != 'early_stopping_rounds'}
    early = XGB_PARAMS.get('early_stopping_rounds', 30)

    model = xgb.XGBRegressor(**params, early_stopping_rounds=early)
    model.fit(
        X_tr, y_train_c,
        eval_set=[(X_te, y_test)],
        verbose=False,
    )

    return {
        'train': {'pred': model.predict(X_tr), 'actual': y_train.values},
        'test':  {'pred': model.predict(X_te), 'actual': y_test.values},
        'val':   {'pred': model.predict(X_va), 'actual': y_val.values},
        'model': model,
    }
```

---

## Step 4 — `evaluate.py` (Scorer)

This is the evaluator ShinkaEvolve calls for every candidate program.
It loads data, calls `run_experiment`, and returns a **composite score** that
penalizes mean prediction collapse, overfitting, and negative out-of-sample signal.

```python
# evolve/evaluate.py

import argparse, importlib.util, os, sys, traceback
import numpy as np
import pandas as pd
from shinka.core import run_shinka_eval

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_PATH = "backtester_v2/data/raw/BTC_USDT_real.parquet"
# Add more tickers here if multi-ticker dataset exists:
# EXTRA_TICKERS = ["backtester_v2/data/raw/ETH_USDT_real.parquet", ...]


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_index()
    return df


def chronological_split(X, y, train_frac=0.8, test_frac=0.1):
    n = len(X)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + test_frac))
    return (
        X.iloc[:t1],  y.iloc[:t1],
        X.iloc[t1:t2], y.iloc[t1:t2],
        X.iloc[t2:],  y.iloc[t2:],
    )


def directional_accuracy(pred, actual):
    mask = actual != 0
    if mask.sum() < 10:
        return 0.5
    return np.mean(np.sign(pred[mask]) == np.sign(actual[mask]))


def compute_score(results: dict) -> tuple[float, dict]:
    """
    Composite fitness score. Higher = better.

    Components:
      1. val_da       — directional accuracy on validation (primary signal)
      2. test_da      — directional accuracy on test set
      3. overfit_pen  — penalty when train DA >> val DA (overfitting)
      4. collapse_pen — penalty when pred std << actual std (mean prediction)
      5. invert_pen   — heavy penalty when val_da < 0.45 (worse than random)

    Score is in [−10, +10] roughly, maximized by ShinkaEvolve.
    """
    train_r = results['train']
    test_r  = results['test']
    val_r   = results['val']

    train_da = directional_accuracy(train_r['pred'], train_r['actual'])
    test_da  = directional_accuracy(test_r['pred'],  test_r['actual'])
    val_da   = directional_accuracy(val_r['pred'],   val_r['actual'])

    # Prediction variance check (collapse penalty)
    pred_std_ratio_val = (
        np.std(val_r['pred']) / (np.std(val_r['actual']) + 1e-9)
    )
    collapse_pen = max(0.0, 0.15 - pred_std_ratio_val) * 5.0
    # If pred_std_ratio < 0.15, apply up to −0.75 penalty

    # Overfitting penalty: gap between train and val DA
    overfit_gap = max(0.0, train_da - val_da - 0.10)
    overfit_pen = overfit_gap * 4.0
    # Penalizes cases where train DA is 10+ points above val DA

    # Inversion penalty: val DA well below random
    invert_pen = 0.0
    if val_da < 0.45:
        invert_pen = (0.45 - val_da) * 20.0
    # E.g. val_da=0.38 → penalty = 1.4

    # Primary score: weighted average of test + val DA, centered at 0.50
    signal_score = (
        (test_da - 0.50) * 3.0 +
        (val_da  - 0.50) * 7.0     # val weighted 70% (unseen regime)
    )

    combined_score = signal_score - collapse_pen - overfit_pen - invert_pen

    public_metrics = {
        "combined_score": round(combined_score, 4),
        "val_da":         round(val_da, 4),
        "test_da":        round(test_da, 4),
        "train_da":       round(train_da, 4),
        "pred_std_ratio": round(pred_std_ratio_val, 4),
        "overfit_gap":    round(train_da - val_da, 4),
    }
    private_metrics = {
        "collapse_penalty": round(collapse_pen, 4),
        "overfit_penalty":  round(overfit_pen, 4),
        "invert_penalty":   round(invert_pen, 4),
    }

    return combined_score, public_metrics, private_metrics


def get_kwargs(run_idx: int) -> dict:
    """Load data once and return as kwargs to run_experiment."""
    df = load_data()

    # Import build_features from the candidate program at runtime.
    # run_shinka_eval injects the candidate module; we access it via a global
    # set in the wrapper below.
    return {"_df": df}


def aggregate_fn(results_list: list) -> dict:
    """
    Called once per candidate with results from all num_runs evaluations.
    We run only 1 evaluation (deterministic), so just unpack index 0.
    """
    res = results_list[0]
    if res is None:
        return {
            "combined_score": -10.0,
            "public": {"error": "run_experiment returned None"},
            "private": {},
        }

    score, public, private = compute_score(res)
    return {
        "combined_score": score,
        "public": public,
        "private": private,
        "text_feedback": _build_text_feedback(public, private),
    }


def _build_text_feedback(pub: dict, priv: dict) -> str:
    lines = [
        f"val_DA={pub['val_da']:.4f}  test_DA={pub['test_da']:.4f}  train_DA={pub['train_da']:.4f}",
        f"pred_std_ratio={pub['pred_std_ratio']:.4f}  overfit_gap={pub['overfit_gap']:.4f}",
        f"penalties → collapse={priv.get('collapse_penalty',0):.3f}  "
        f"overfit={priv.get('overfit_penalty',0):.3f}  "
        f"invert={priv.get('invert_penalty',0):.3f}",
        f"SCORE={pub['combined_score']:.4f}",
    ]
    if pub['val_da'] < 0.45:
        lines.append("WARNING: model is inverse-correlated on validation — penalized heavily.")
    if pub['pred_std_ratio'] < 0.15:
        lines.append("WARNING: predictions collapsed toward mean — increase model expressivity or remove regularization.")
    if pub['overfit_gap'] > 0.10:
        lines.append("WARNING: overfitting detected — increase regularization or reduce depth.")
    return "\n".join(lines)


# ── Shinka wrapper ─────────────────────────────────────────────────────────────

def _make_run_fn(program_path: str):
    """Dynamically load the candidate program and expose run_experiment."""
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run_fn(_df: pd.DataFrame):
        df = _df
        X_all = mod.build_features(df)
        y_all = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
        y_all = y_all.loc[X_all.index].dropna()
        X_all = X_all.loc[y_all.index]

        X_tr, y_tr, X_te, y_te, X_va, y_va = chronological_split(X_all, y_all)
        return mod.run_experiment(X_tr, y_tr, X_te, y_te, X_va, y_va)

    return run_fn


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    try:
        run_fn = _make_run_fn(program_path)
    except Exception as e:
        print(f"LOAD ERROR: {e}\n{traceback.format_exc()}")
        _write_failure(results_dir, str(e))
        return

    df = load_data()

    try:
        result = run_fn(df)
    except Exception as e:
        print(f"RUN ERROR: {e}\n{traceback.format_exc()}")
        _write_failure(results_dir, str(e))
        return

    score, public, private = compute_score(result)
    fb = _build_text_feedback(public, private)

    # Write shinka-compatible output
    import json
    out = {
        "combined_score": score,
        "public": public,
        "private": private,
        "text_feedback": fb,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(fb)
    print(f"\nFinal score: {score:.4f}")


def _write_failure(results_dir, msg):
    import json
    out = {"combined_score": -10.0, "public": {"error": msg}, "private": {}}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir",  required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
```

---

## Step 5 — `run_evolution.py` (Launcher)

```python
# evolve/run_evolution.py

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(
    eval_program_path="evolve/evaluate.py",
    conda_env=None,              # set to your venv name if using conda
)

db_config = DatabaseConfig(
    num_islands=4,               # 4 parallel evolutionary islands
    archive_size=50,             # keep top-50 programs per island
    elite_selection_ratio=0.3,
    num_archive_inspirations=5,
    migration_interval=10,
    parent_selection_strategy="power_law",
)

evo_config = EvolutionConfig(
    init_program_path="evolve/initial.py",
    num_generations=100,
    max_parallel_jobs=4,         # adjust to your CPU count
    llm_models=["claude-sonnet-4-20250514"],   # or gpt-4.1-mini for cost
    use_text_feedback=True,      # feed the penalty breakdown back to LLM
    patch_types=["diff", "full"],
    patch_type_probs=[0.8, 0.2],
    task_sys_msg="""
You are optimizing an XGBoost trading model for 15-minute BTC/ETH/SOL perp futures.

The goal is to maximize out-of-sample directional accuracy (DA) on unseen validation data.
DA = fraction of bars where sign(predicted BPS) == sign(actual BPS).
Random baseline is 0.50. Target: val_DA >= 0.52 consistently.

You may modify:
  1. XGB_PARAMS — hyperparameters (depth, estimators, regularization, learning rate, etc.)
  2. build_features() — which features to compute, lookback windows, transformations,
     regime indicators, feature interactions, normalization methods.
  3. CLIP_PERCENTILE and MIN_PRED_STD_RATIO constants.

Constraints:
  - build_features() must use ONLY past data (shift by at least 1 before use)
  - Do not modify run_experiment() or the return format
  - Do not hardcode split indices or peek at test/val targets
  - Keep total feature count <= 80 to avoid curse of dimensionality
  - Execution time per candidate must stay under 3 minutes

Common failure modes to avoid:
  - pred_std_ratio < 0.15: model is predicting the mean → loosen regularization or add signal features
  - overfit_gap > 0.10: train DA >> val DA → increase regularization, reduce depth, fewer estimators
  - val_DA < 0.45: model is inverted on validation → add regime-conditioning, regime features, or shorten lookbacks
""",
)

runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)

if __name__ == "__main__":
    runner.run()
```

---

## Step 6 — Score Formula Reference

The evaluator combines the following into `combined_score` (maximized by Shinka):

```
signal_score  =  (test_DA − 0.50) × 3.0  +  (val_DA − 0.50) × 7.0

penalties:
  collapse_pen  =  max(0,  0.15 − pred_std_ratio_val)  × 5.0
  overfit_pen   =  max(0,  train_DA − val_DA − 0.10)   × 4.0
  invert_pen    =  max(0,  0.45 − val_DA)               × 20.0

combined_score  =  signal_score − collapse_pen − overfit_pen − invert_pen
```

| Scenario | Score |
|---|---|
| val_DA=0.53, test_DA=0.52, no penalties | **+0.42** |
| val_DA=0.50 (random), no penalties | **0.00** |
| val_DA=0.38 (inverted) | **−1.40** (invert_pen alone) |
| pred_std_ratio=0.05 (collapsed) | **−0.50** (collapse_pen alone) |
| train_DA=0.60, val_DA=0.48 (overfit+invert) | **−1.08 − 0.14 = −1.22** |

---

## Step 7 — Running

```bash
# From project root
cd ShinkaEvolve
source .venv/bin/activate

# Launch evolution
python evolve/run_evolution.py

# Monitor in real time (separate terminal)
shinka_visualize --port 8888 --open
```

The WebUI shows the genealogy tree, score history per island, and the code diff of each
evolved candidate. The best program found across all generations is saved automatically
to `results/best_program.py`.

---

## Step 8 — Extracting the Best Model

After evolution completes (or at any point during), extract and retrain the winner:

```bash
# Copy best evolved program
cp results/<run_id>/best_program.py backtester_v2/scripts/evolved_model.py

# Retrain on full train+test data, evaluate on val
python -c "
import pandas as pd, numpy as np
import importlib.util, sys

spec = importlib.util.spec_from_file_location('best', 'backtester_v2/scripts/evolved_model.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df = pd.read_parquet('backtester_v2/data/raw/BTC_USDT_real.parquet')
X = mod.build_features(df)
y = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
y = y.loc[X.index].dropna()
X = X.loc[y.index]

n = len(X)
X_val, y_val = X.iloc[int(n*0.9):], y.iloc[int(n*0.9):]
result = mod.run_experiment(X.iloc[:int(n*0.9)], y.iloc[:int(n*0.9)],
                             X_val, y_val, X_val, y_val)
da = np.mean(np.sign(result['val']['pred']) == np.sign(result['val']['actual']))
print(f'Final val DA: {da:.4f}')
"
```

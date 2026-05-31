# evolve/evaluate.py

import argparse, importlib.util, os, sys, traceback
import json
import numpy as np
import pandas as pd

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


def pearson_corr(pred, actual):
    if len(pred) < 2 or np.std(pred) == 0 or np.std(actual) == 0:
        return 0.0
    corr = np.corrcoef(actual, pred)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def regression_slope(pred, actual):
    if len(pred) < 2 or np.std(actual) == 0:
        return 0.0
    slope = np.polyfit(actual, pred, 1)[0]
    return float(slope) if np.isfinite(slope) else 0.0


def compute_score(results: dict) -> tuple[float, dict, dict]:
    """
    Composite fitness score. Higher = better.

    Components:
      1. val_da       — directional accuracy on validation (primary signal)
      2. test_da      — directional accuracy on test set
      3. collapse_pen — penalty when pred std << actual std (mean prediction)
      4. slope_pen    — penalty when predictive slope disappears on validation
      5. gen_pen      — penalty when validation correlation collapses vs test
      6. overfit_pen  — penalty when train DA >> val DA (overfitting)
      7. invert_pen   — heavy penalty when val_da < 0.45 (worse than random)

    Score is in [−10, +10] roughly, maximized by ShinkaEvolve.
    """
    train_r = results['train']
    test_r  = results['test']
    val_r   = results['val']

    train_da = directional_accuracy(train_r['pred'], train_r['actual'])
    test_da  = directional_accuracy(test_r['pred'],  test_r['actual'])
    val_da   = directional_accuracy(val_r['pred'],   val_r['actual'])

    train_slope = regression_slope(train_r['pred'], train_r['actual'])
    test_slope = regression_slope(test_r['pred'], test_r['actual'])
    val_slope = regression_slope(val_r['pred'], val_r['actual'])
    slope_ratio = val_slope / (train_slope + 1e-9)

    train_pearson = pearson_corr(train_r['pred'], train_r['actual'])
    test_pearson = pearson_corr(test_r['pred'], test_r['actual'])
    val_pearson = pearson_corr(val_r['pred'], val_r['actual'])
    generalization_ratio = val_pearson / (abs(test_pearson) + 1e-9)

    # Prediction variance check (collapse penalty)
    pred_std_ratio_val = (
        np.std(val_r['pred']) / (np.std(val_r['actual']) + 1e-9)
    )
    collapse_pen = max(0.0, 0.20 - pred_std_ratio_val) * 8.0

    # Penalize models where train slope does not carry into validation.
    slope_pen = max(0.0, 0.30 - slope_ratio) * 6.0

    # Penalize models whose validation correlation collapses vs test.
    gen_pen = max(0.0, 0.40 - generalization_ratio) * 5.0

    # Inversion penalty: val DA well below random.
    invert_pen = max(0.0, 0.45 - val_da) * 20.0

    # Overfitting penalty: tighten tolerated train→val DA gap.
    overfit_gap = max(0.0, train_da - val_da - 0.08)
    overfit_pen = overfit_gap * 5.0

    # Primary score: weighted average of test + val DA, centered at 0.50
    signal_score = (
        (test_da - 0.50) * 3.0 +
        (val_da  - 0.50) * 7.0     # val weighted 70% (unseen regime)
    )

    combined_score = (
        signal_score
        - collapse_pen
        - slope_pen
        - gen_pen
        - invert_pen
        - overfit_pen
    )

    public_metrics = {
        "combined_score":       round(combined_score, 4),
        "val_da":               round(val_da, 4),
        "test_da":              round(test_da, 4),
        "train_da":             round(train_da, 4),
        "train_pearson":        round(train_pearson, 4),
        "test_pearson":         round(test_pearson, 4),
        "val_pearson":          round(val_pearson, 4),
        "train_slope":          round(train_slope, 4),
        "test_slope":           round(test_slope, 4),
        "val_slope":            round(val_slope, 4),
        "slope_ratio":          round(slope_ratio, 4),
        "generalization_ratio": round(generalization_ratio, 4),
        "pred_std_ratio":       round(pred_std_ratio_val, 4),
        "overfit_gap":          round(train_da - val_da, 4),
    }
    private_metrics = {
        "collapse_penalty": round(collapse_pen, 4),
        "slope_penalty":    round(slope_pen, 4),
        "gen_penalty":      round(gen_pen, 4),
        "overfit_penalty":  round(overfit_pen, 4),
        "invert_penalty":   round(invert_pen, 4),
        "signal_score":     round(signal_score, 4),
    }

    return combined_score, public_metrics, private_metrics


def _build_text_feedback(pub: dict, priv: dict) -> str:
    lines = [
        f"val_DA={pub['val_da']:.4f}  test_DA={pub['test_da']:.4f}  train_DA={pub['train_da']:.4f}",
        f"val_r={pub['val_pearson']:.4f}  test_r={pub['test_pearson']:.4f}  train_r={pub['train_pearson']:.4f}",
        f"val_slope={pub['val_slope']:.4f}  train_slope={pub['train_slope']:.4f}  slope_ratio={pub['slope_ratio']:.1%}",
        f"pred_std_ratio={pub['pred_std_ratio']:.4f}  generalization_ratio={pub['generalization_ratio']:.1%}  overfit_gap={pub['overfit_gap']:.4f}",
        f"penalties → collapse={priv.get('collapse_penalty',0):.3f}  "
        f"slope={priv.get('slope_penalty',0):.3f}  "
        f"gen={priv.get('gen_penalty',0):.3f}  "
        f"overfit={priv.get('overfit_penalty',0):.3f}  "
        f"invert={priv.get('invert_penalty',0):.3f}",
        f"SCORE={pub['combined_score']:.4f}",
    ]
    if pub['val_da'] < 0.45:
        lines.append("WARNING: model is inverse-correlated on validation — penalized heavily.")
    if pub['pred_std_ratio'] < 0.20:
        lines.append("WARNING: predictions collapsed toward mean — increase model expressivity or remove regularization.")
    if pub['overfit_gap'] > 0.08:
        lines.append("WARNING: overfitting detected — increase regularization or reduce depth.")
    if pub['slope_ratio'] < 0.30:
        lines.append(
            f"WARNING: val slope ({pub['val_slope']:.4f}) is only "
            f"{pub['slope_ratio']:.1%} of train slope ({pub['train_slope']:.4f}). "
            "Features are regime-specific. Try shorter lookbacks, z-score normalization "
            "within rolling windows, or regime-conditional feature scaling."
        )
    if pub['generalization_ratio'] < 0.40:
        lines.append(
            f"WARNING: val_r is only {pub['generalization_ratio']:.1%} of test_r. "
            "The model generalizes across the train→test boundary but breaks at the "
            "test→val boundary. This is a second-order regime shift. Add features "
            "that measure the CHANGE in market character, not just the current state."
        )
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

    try:
        score, public, private = compute_score(result)
    except Exception as e:
        print(f"SCORE ERROR: {e}\n{traceback.format_exc()}")
        _write_failure(results_dir, str(e))
        return

    if not np.isfinite(score):
        msg = f"combined_score is not finite: {score}"
        print(f"SCORE ERROR: {msg}")
        _write_failure(results_dir, msg)
        return

    fb = _build_text_feedback(public, private)

    # Write shinka-compatible output
    metrics = {
        "combined_score": score,
        "public": public,
        "private": private,
        "text_feedback": fb,
    }
    _write_results(results_dir, metrics, correct=True, error=None)

    print(fb)
    print(f"\nFinal score: {score:.4f}")


def _write_results(results_dir, metrics, correct, error):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": bool(correct), "error": error}, f, indent=2)


def _write_failure(results_dir, msg):
    metrics = {
        "combined_score": -10.0,
        "public": {"error": msg},
        "private": {},
        "text_feedback": msg,
    }
    _write_results(results_dir, metrics, correct=False, error=msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir",  required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)

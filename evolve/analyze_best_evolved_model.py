import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("evolve/results/mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "backtester_v2/data/raw/BTC_USDT_real.parquet"
DEFAULT_PROGRAM_PATH = ROOT / "results_20260511_211813/best/main.py"
DEFAULT_OUTPUT_DIR = ROOT / "backtester_v2/results"


def load_candidate(program_path: Path):
    spec = importlib.util.spec_from_file_location("best_evolved_candidate", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evolved program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chronological_split(X: pd.DataFrame, y: pd.Series):
    total = len(X)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)
    return {
        "train": (X.iloc[:train_end], y.iloc[:train_end]),
        "test": (X.iloc[train_end:test_end], y.iloc[train_end:test_end]),
        "val": (X.iloc[test_end:], y.iloc[test_end:]),
    }


def directional_accuracy(pred_bps: np.ndarray, actual_bps: np.ndarray) -> float:
    mask = actual_bps != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(pred_bps[mask]) == np.sign(actual_bps[mask])))


def regression_stats(actual_ratio: np.ndarray, predicted_ratio: np.ndarray) -> dict:
    actual_bps = (actual_ratio - 1.0) * 10000.0
    pred_bps = (predicted_ratio - 1.0) * 10000.0
    if len(actual_ratio) > 1 and np.std(actual_ratio) > 0 and np.std(predicted_ratio) > 0:
        corr = float(np.corrcoef(actual_ratio, predicted_ratio)[0, 1])
        slope, intercept = np.polyfit(actual_ratio, predicted_ratio, 1)
    else:
        corr = float("nan")
        slope = float("nan")
        intercept = float("nan")

    rmse = float(np.sqrt(mean_squared_error(actual_ratio, predicted_ratio)))
    rmse_bps = float(np.sqrt(mean_squared_error(actual_bps, pred_bps)))

    return {
        "n": int(len(actual_ratio)),
        "r2": float(r2_score(actual_ratio, predicted_ratio)),
        "pearson_corr": corr,
        "rmse_ratio": rmse,
        "mae_ratio": float(mean_absolute_error(actual_ratio, predicted_ratio)),
        "rmse_bps": rmse_bps,
        "mae_bps": float(mean_absolute_error(actual_bps, pred_bps)),
        "bias_bps": float(np.mean(pred_bps - actual_bps)),
        "slope": float(slope),
        "intercept": float(intercept),
        "directional_accuracy": directional_accuracy(pred_bps, actual_bps),
        "actual_mean_ratio": float(np.mean(actual_ratio)),
        "predicted_mean_ratio": float(np.mean(predicted_ratio)),
        "actual_std_bps": float(np.std(actual_bps)),
        "predicted_std_bps": float(np.std(pred_bps)),
        "predicted_to_actual_std_ratio": float(np.std(pred_bps) / (np.std(actual_bps) + 1e-12)),
    }


def make_scatter_plot(predictions: pd.DataFrame, stats: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"train": "#1f77b4", "test": "#ff7f0e", "val": "#2ca02c"}

    for split, group in predictions.groupby("split", sort=False):
        sample = group
        if len(sample) > 12000:
            sample = sample.sample(12000, random_state=42)
        ax.scatter(
            sample["actual_ratio"],
            sample["predicted_ratio"],
            s=5,
            alpha=0.28,
            label=f"{split} (n={len(group):,}, r={stats[split]['pearson_corr']:.3f})",
            color=colors.get(split),
            linewidths=0,
        )

    min_lim = min(predictions["actual_ratio"].min(), predictions["predicted_ratio"].min())
    max_lim = max(predictions["actual_ratio"].max(), predictions["predicted_ratio"].max())
    padding = (max_lim - min_lim) * 0.05
    lims = [min_lim - padding, max_lim + padding]
    ax.plot(lims, lims, "k--", linewidth=1.1, alpha=0.75, label="Parity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Best ShinkaEvolve Model: Predicted vs Actual Next-Bar Ratios")
    ax.set_xlabel("Actual ratio")
    ax.set_ylabel("Predicted ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(stats: dict, output_path: Path, program_path: Path, plot_path: Path):
    columns = [
        "split",
        "n",
        "r2",
        "pearson_corr",
        "rmse_bps",
        "mae_bps",
        "bias_bps",
        "slope",
        "directional_accuracy",
        "predicted_to_actual_std_ratio",
    ]
    rows = []
    for split in ["train", "test", "val", "all"]:
        row = {"split": split}
        row.update(stats[split])
        rows.append(row)

    table = pd.DataFrame(rows)[columns]
    report = [
        "# Best ShinkaEvolve Regression Analysis",
        "",
        f"Evolved program: `{program_path.relative_to(ROOT)}`",
        f"Scatter plot: `{plot_path.relative_to(ROOT)}`",
        "",
        table.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    output_path.write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--program-path", type=Path, default=DEFAULT_PROGRAM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    module = load_candidate(args.program_path)
    df = pd.read_parquet(args.data_path).sort_index()
    X = module.build_features(df)
    y_bps = (df["close"].shift(-1) / df["close"] - 1.0) * 10000.0
    y_bps = y_bps.loc[X.index].dropna()
    X = X.loc[y_bps.index]

    splits = chronological_split(X, y_bps)
    result = module.run_experiment(
        splits["train"][0],
        splits["train"][1],
        splits["test"][0],
        splits["test"][1],
        splits["val"][0],
        splits["val"][1],
    )

    frames = []
    stats = {}
    for split in ["train", "test", "val"]:
        actual_bps = np.asarray(result[split]["actual"], dtype=float)
        pred_bps = np.asarray(result[split]["pred"], dtype=float)
        frame = pd.DataFrame(
            {
                "split": split,
                "actual_bps": actual_bps,
                "predicted_bps": pred_bps,
                "actual_ratio": 1.0 + actual_bps / 10000.0,
                "predicted_ratio": 1.0 + pred_bps / 10000.0,
            },
            index=splits[split][1].index,
        )
        frames.append(frame)
        stats[split] = regression_stats(
            frame["actual_ratio"].to_numpy(),
            frame["predicted_ratio"].to_numpy(),
        )

    predictions = pd.concat(frames)
    stats["all"] = regression_stats(
        predictions["actual_ratio"].to_numpy(),
        predictions["predicted_ratio"].to_numpy(),
    )

    predictions_path = output_dir / "best_evolved_ratio_predictions.csv"
    stats_json_path = output_dir / "best_evolved_regression_stats.json"
    stats_csv_path = output_dir / "best_evolved_regression_stats.csv"
    plot_path = output_dir / "best_evolved_ratio_scatter.png"
    report_path = output_dir / "best_evolved_regression_report.md"

    predictions.to_csv(predictions_path, index_label="timestamp")
    stats_json_path.write_text(json.dumps(stats, indent=2))
    pd.DataFrame([{"split": k, **v} for k, v in stats.items()]).to_csv(stats_csv_path, index=False)
    make_scatter_plot(predictions, stats, plot_path)
    write_markdown_report(stats, report_path, args.program_path, plot_path)

    print(f"Wrote {plot_path}")
    print(f"Wrote {stats_csv_path}")
    print(f"Wrote {stats_json_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {report_path}")
    print(pd.DataFrame([{'split': k, **v} for k, v in stats.items()])[
        ["split", "n", "r2", "pearson_corr", "rmse_bps", "mae_bps", "bias_bps", "directional_accuracy"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.getcwd())

from backtester_v2.scripts.feature_engineering import build_features

def main():
    data_dir = 'backtester_v2/data/raw'
    results_dir = 'backtester_v2/results'
    os.makedirs(results_dir, exist_ok=True)

    symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
    all_data = []

    print("Loading and processing features for each ticker...")
    for symbol in symbols:
        path = f"{data_dir}/{symbol}_real.parquet"
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        X_sym = build_features(df)
        y_bps = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
        data = X_sym.copy()
        data['target'] = y_bps.loc[X_sym.index]
        data = data.dropna(subset=['target'])
        all_data.append(data)

    if not all_data:
        print("No data loaded.")
        return

    # Find common time range
    min_ts = max(d.index.min() for d in all_data)
    max_ts = min(d.index.max() for d in all_data)
    filtered_data = [d[(d.index >= min_ts) & (d.index <= max_ts)] for d in all_data]
    combined = pd.concat(filtered_data).sort_index()
    unique_ts = combined.index.unique().sort_values()
    total_steps = len(unique_ts)
    
    # 80/10/10 time-based split
    train_end_ts = unique_ts[int(total_steps * 0.8)]
    test_end_ts = unique_ts[int(total_steps * 0.9)]

    train_set = combined[combined.index <= train_end_ts]
    test_set = combined[(combined.index > train_end_ts) & (combined.index <= test_end_ts)]
    val_set = combined[combined.index > test_end_ts]

    # STEP 4: CLIP OUTLIERS
    print("Step 4: Clipping target outliers (1st-99th percentile)...")
    lower_q = train_set['target'].quantile(0.01)
    upper_q = train_set['target'].quantile(0.99)
    y_train = train_set['target'].clip(lower=lower_q, upper=upper_q)
    print(f"  Target range clipped to: [{lower_q:.2f}, {upper_q:.2f}]")
    
    X_train = train_set.drop(columns=['target'])
    X_test = test_set.drop(columns=['target'])
    y_test = test_set['target']
    X_val = val_set.drop(columns=['target'])
    y_val = val_set['target']

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Constrained Hyperparameters (Step 3 refinement)
    XGB_PARAMS = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.6,
        "learning_rate": 0.05, # Increased LR for better gradient signal
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "hist"
    }

    print("Step 3: Training model with BPS targets...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    # Predictions (Raw, no calibration)
    y_train_raw = model.predict(X_train_scaled)
    y_test_raw = model.predict(X_test_scaled)
    y_val_raw = model.predict(X_val_scaled)

    # STEP 1: DIAGNOSE VARIANCE
    def print_stats(split_name, actual, raw):
        print(f"\n--- Statistics for {split_name} ---")
        print(f"std of actual BPS: {actual.std():.4f}")
        print(f"std of raw model outputs: {raw.std():.4f}")
        da = np.mean(np.sign(actual) == np.sign(raw))
        print(f"Directional Accuracy: {da:.4f}")
        return da

    print_stats("TRAIN", y_train, y_train_raw)
    da_test = print_stats("TEST", y_test, y_test_raw)
    da_val = print_stats("VAL", y_val, y_val_raw)

    # Histogram of raw outputs
    plt.figure(figsize=(10, 6))
    plt.hist(y_train_raw, bins=50, alpha=0.5, label='Train Raw', color='blue')
    plt.hist(y_test_raw, bins=50, alpha=0.5, label='Test Raw', color='green')
    plt.title("Histogram of Raw XGBoost BPS Outputs")
    plt.xlabel("Predicted BPS")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "raw_output_histogram.png"))
    plt.close()

    # STEP 6: REPLOT SCATTERS IN BPS SPACE
    def plot_bps_scatter(actual, predicted, title, filename):
        corr = np.corrcoef(actual, predicted)[0, 1]
        plt.figure(figsize=(10, 10))
        plt.scatter(actual, predicted, alpha=0.2, s=2, color='purple')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        
        # Add parity line
        lims = [-100, 100] # Standard BPS zoom
        plt.plot(lims, lims, 'r--', alpha=0.7, label='y=x')
        
        plt.xlim([-50, 50])
        plt.ylim([-20, 20]) # Compressed y as predictions are smaller
        
        plt.title(f"{title}\nCorr: {corr:.4f} | Zoomed View")
        plt.xlabel("Actual BPS")
        plt.ylabel("Predicted BPS")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    print("\nStep 6: Generating BPS scatter plots...")
    plot_bps_scatter(y_train, y_train_raw, "Robust Model: Train BPS", os.path.join(results_dir, "bps_scatter_train.png"))
    plot_bps_scatter(y_test, y_test_raw, "Robust Model: Test BPS", os.path.join(results_dir, "bps_scatter_test.png"))
    plot_bps_scatter(y_val, y_val_raw, "Robust Model: Val BPS", os.path.join(results_dir, "bps_scatter_val.png"))

    # Verdict
    print(f"\nFinal Verdict:")
    if da_test > 0.52 and da_val > 0.52:
        print("SUCCESS: Directional Accuracy > 0.52 on both test and val!")
    else:
        print("FAILURE: Edge (DA) is still insufficient. Adjusting might be needed.")

if __name__ == "__main__":
    main()

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

def plot_scatter(actual, predicted, title, filename):
    corr = np.corrcoef(actual, predicted)[0, 1]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(actual, predicted, alpha=0.1, s=1, color='purple')
    
    # Add parity line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label='Parity (y=x)')
    
    plt.title(f"{title}\nCorrelation: {corr:.4f}")
    plt.xlabel("Actual Price Ratio")
    plt.ylabel("Predicted Price Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename} (Corr: {corr:.4f})")
    return corr

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
        y_sym = y_bps.loc[X_sym.index]
        
        data = X_sym.copy()
        data['target'] = y_sym
        data = data.dropna(subset=['target'])
        all_data.append(data)

    if not all_data:
        print("No data loaded.")
        return

    # Find common time range
    min_ts = max(d.index.min() for d in all_data)
    max_ts = min(d.index.max() for d in all_data)
    
    # Filter and Combine
    filtered_data = []
    for d in all_data:
        filtered_data.append(d[(d.index >= min_ts) & (d.index <= max_ts)])
    
    combined = pd.concat(filtered_data).sort_index()
    unique_ts = combined.index.unique().sort_values()
    total_steps = len(unique_ts)
    
    # 80/10/10 time-based split
    train_end_ts = unique_ts[int(total_steps * 0.8)]
    test_end_ts = unique_ts[int(total_steps * 0.9)]

    train_set = combined[combined.index <= train_end_ts]
    test_set = combined[(combined.index > train_end_ts) & (combined.index <= test_end_ts)]
    val_set = combined[combined.index > test_end_ts]

    X_train = train_set.drop(columns=['target'])
    y_train = train_set['target']
    X_test = test_set.drop(columns=['target'])
    y_test = test_set['target']
    X_val = val_set.drop(columns=['target'])
    y_val = val_set['target']

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Constrained XGBoost Model
    XGB_PARAMS = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.6,
        "learning_rate": 0.01,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "hist"
    }

    print("Training robust multi-ticker model...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    # Predict
    print("Generating predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Convert to Ratios for plotting
    y_train_actual_ratio = 1.0 + (y_train / 10000.0)
    y_train_pred_ratio = 1.0 + (y_train_pred / 10000.0)
    
    y_test_actual_ratio = 1.0 + (y_test / 10000.0)
    y_test_pred_ratio = 1.0 + (y_test_pred / 10000.0)
    
    y_val_actual_ratio = 1.0 + (y_val / 10000.0)
    y_val_pred_ratio = 1.0 + (y_val_pred / 10000.0)

    print("Generating plots and exporting data...")
    corr_train = plot_scatter(y_train_actual_ratio, y_train_pred_ratio, "Robust Model: Train Set (BTC/ETH/SOL)", 
                 os.path.join(results_dir, "robust_scatter_train.png"))
    corr_test = plot_scatter(y_test_actual_ratio, y_test_pred_ratio, "Robust Model: Test Set (BTC/ETH/SOL)", 
                 os.path.join(results_dir, "robust_scatter_test.png"))
    corr_val = plot_scatter(y_val_actual_ratio, y_val_pred_ratio, "Robust Model: Validation Set (BTC/ETH/SOL)", 
                 os.path.join(results_dir, "robust_scatter_val.png"))

    # 6. Save Data to CSV
    train_df = pd.DataFrame({
        'timestamp': train_set.index,
        'symbol': train_set['symbol'] if 'symbol' in train_set else 'unknown',
        'actual_ratio': y_train_actual_ratio,
        'predicted_ratio': y_train_pred_ratio,
        'split': 'train'
    })
    test_df = pd.DataFrame({
        'timestamp': test_set.index,
        'symbol': test_set['symbol'] if 'symbol' in test_set else 'unknown',
        'actual_ratio': y_test_actual_ratio,
        'predicted_ratio': y_test_pred_ratio,
        'split': 'test'
    })
    val_df = pd.DataFrame({
        'timestamp': val_set.index,
        'symbol': val_set['symbol'] if 'symbol' in val_set else 'unknown',
        'actual_ratio': y_val_actual_ratio,
        'predicted_ratio': y_val_pred_ratio,
        'split': 'validation'
    })
    
    all_preds = pd.concat([train_df, test_df, val_df])
    csv_path = os.path.join(results_dir, "robust_model_predictions.csv")
    all_preds.to_csv(csv_path, index=False)
    print(f"Saved prediction data to {csv_path}")

    # 7. Generate Markdown Report
    report_path = os.path.join(results_dir, "robust_model_report.md")
    with open(report_path, "w") as f:
        f.write("# Robust Multi-Ticker XGBoost Model Report\n\n")
        f.write("## Model Details (Hyperparameters)\n")
        f.write("```json\n")
        import json
        f.write(json.dumps(XGB_PARAMS, indent=4))
        f.write("\n```\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Dataset | Correlation | Status |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Training Set** | {corr_train:.4f} | In-sample fit |\n")
        f.write(f"| **Testing Set** | {corr_test:.4f} | {'BENCHMARK HIT (>= 0.05)' if corr_test >= 0.05 else 'Below benchmark'} |\n")
        f.write(f"| **Validation Set** | {corr_val:.4f} | Out-of-sample evaluation |\n\n")
        
        f.write("## Observations\n")
        f.write("- **Data Sources**: Real Kraken Futures OHLCV/Funding + Real Binance Vision Depth.\n")
        f.write("- **Tickers**: BTC_USDT, ETH_USDT, SOL_USDT.\n")
        f.write("- **Split Method**: Strict chronological (Time-series) 80/10/10.\n")
        f.write("- **Refinement**: Model complexity reduced to improve generalization.\n")
        
    print(f"Saved markdown report to {report_path}")

if __name__ == "__main__":
    main()

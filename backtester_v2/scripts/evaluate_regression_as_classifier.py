import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import sys

# Add project root to path to import feature_engineering
sys.path.append(os.getcwd())

from backtester_v2.scripts.feature_engineering import build_features

def main():
    # Paths
    data_path = 'backtester_v2/data/raw/BTC_USDT_50k.parquet'
    model_path = 'backtester_v1/models/xgb_regression_v1.json'
    scaler_path = 'backtester_v1/models/scaler_v1.joblib'
    calibration_path = 'backtester_v1/models/calibration_v1.joblib'
    results_dir = 'backtester_v2/results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Please run download_data.py first.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Build Features
    print("Building features...")
    # build_features drops NaNs (first ~60 rows)
    X = build_features(df)
    
    # Target: Next bar BPS change (Ground Truth)
    # Target: (close_{t+1} / close_t - 1.0) * 10000
    y_bps = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
    y_actual = y_bps.loc[X.index]
    
    # Drop rows where target is NaN (the very last row usually)
    valid_idx = y_actual.dropna().index
    X = X.loc[valid_idx]
    y_actual = y_actual.loc[valid_idx]
    
    print(f"Testing on {len(X)} samples.")

    # 3. Load Model and Scaler
    print("Loading model and scaler...")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Optional: Calibration factor
    calibration_factor = 1.0
    if os.path.exists(calibration_path):
        calibration_factor = joblib.load(calibration_path)
        print(f"Loaded calibration factor: {calibration_factor}")

    # 4. Scale and Predict
    print("Predicting...")
    X_scaled = scaler.transform(X)
    y_pred_bps = model.predict(X_scaled) * calibration_factor
    
    # 5. Convert to Binary Classes
    # User instruction: Ratio > 1 is 1, Ratio < 1 is 0.
    # BPS = (Ratio - 1) * 10000. So Ratio > 1 <=> BPS > 0.
    y_pred_bin = (y_pred_bps > 0).astype(int)
    y_actual_bin = (y_actual > 0).astype(int)
    
    # 6. Evaluation
    acc = accuracy_score(y_actual_bin, y_pred_bin)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_actual_bin, y_pred_bin)
    print(report)
    
    # Save report to file
    with open(os.path.join(results_dir, 'regression_as_classifier_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 7. Generate Plots
    
    # Heat Map (Confusion Matrix)
    cm = confusion_matrix(y_actual_bin, y_pred_bin)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix\n(Accuracy: {acc:.4f})')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(results_dir, 'regression_as_classifier_heatmap.png'))
    print(f"Saved heatmap to {results_dir}/regression_as_classifier_heatmap.png")
    
    # Scatter Plot (Predicted BPS vs Actual BPS)
    # This helps see the relationship and thresholding
    plt.figure(figsize=(10, 8))
    plt.scatter(y_actual, y_pred_bps, alpha=0.1, s=1)
    
    # Add quadrants lines
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Add a diagonal line for reference (y=x)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    plt.title('Predicted BPS vs Actual BPS\n(Quadrants represent classification accuracy)')
    plt.xlabel('Actual BPS (Ratio-1)*10000')
    plt.ylabel('Predicted BPS (Ratio-1)*10000')
    plt.grid(True, alpha=0.3)
    
    # Annotate quadrants
    plt.text(plt.xlim()[1]*0.7, plt.ylim()[1]*0.7, 'True Positive', fontsize=12, color='green')
    plt.text(plt.xlim()[0]*0.7, plt.ylim()[0]*0.7, 'True Negative', fontsize=12, color='green')
    plt.text(plt.xlim()[0]*0.7, plt.ylim()[1]*0.7, 'False Positive', fontsize=12, color='red')
    plt.text(plt.xlim()[1]*0.7, plt.ylim()[0]*0.7, 'False Negative', fontsize=12, color='red')
    
    plt.savefig(os.path.join(results_dir, 'regression_as_classifier_scatter.png'))
    print(f"Saved scatter plot to {results_dir}/regression_as_classifier_scatter.png")

if __name__ == "__main__":
    main()

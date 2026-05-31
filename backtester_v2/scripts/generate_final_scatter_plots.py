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
from backtester_v1.scripts.backtester import predict_ratios

def plot_scatter(actual, predicted, title, filename):
    corr = np.corrcoef(actual, predicted)[0, 1]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(actual, predicted, alpha=0.2, s=2, color='blue')
    
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
    # Paths
    data_path = 'backtester_v2/data/raw/BTC_USDT_real.parquet'
    model_path = 'backtester_v2/models/xgb_regression_final.json'
    scaler_path = 'backtester_v2/models/scaler_final.joblib'
    calib_path = 'backtester_v2/models/calibration_final.joblib'
    results_dir = 'backtester_v2/results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data and Assets
    print("Loading data and model assets...")
    df = pd.read_parquet(data_path)
    model = xgb.Booster()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    calibration_factor = joblib.load(calib_path)

    # 2. Reconstruct Features and Targets
    print("Building features...")
    X = build_features(df)
    y_bps = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
    y = y_bps.loc[X.index].dropna()
    X = X.loc[y.index]
    
    # 3. Split data (must match train_final_model.py exactly)
    total = len(X)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)
    
    X_train = X.iloc[:train_end]
    X_test = X.iloc[train_end:test_end]
    X_val = X.iloc[test_end:]
    
    y_train_actual_ratio = 1.0 + (y.iloc[:train_end] / 10000.0)
    y_test_actual_ratio = 1.0 + (y.iloc[train_end:test_end] / 10000.0)
    y_val_actual_ratio = 1.0 + (y.iloc[test_end:] / 10000.0)

    # 4. Predict
    print("Generating predictions...")
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)

    y_train_pred_ratio = predict_ratios(model, X_train_scaled, calibration_factor)
    y_test_pred_ratio = predict_ratios(model, X_test_scaled, calibration_factor)
    y_val_pred_ratio = predict_ratios(model, X_val_scaled, calibration_factor)

    # 5. Plot
    print("Generating plots...")
    plot_scatter(y_train_actual_ratio, y_train_pred_ratio, "Train Set: Actual vs Predicted Ratio", 
                 os.path.join(results_dir, "final_scatter_train.png"))
    plot_scatter(y_test_actual_ratio, y_test_pred_ratio, "Test Set: Actual vs Predicted Ratio", 
                 os.path.join(results_dir, "final_scatter_test.png"))
    plot_scatter(y_val_actual_ratio, y_val_pred_ratio, "Validation Set: Actual vs Predicted Ratio", 
                 os.path.join(results_dir, "final_scatter_val.png"))

if __name__ == "__main__":
    main()

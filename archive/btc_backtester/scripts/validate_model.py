import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
import joblib

# Add the project root to sys.path
sys.path.append(os.path.join(os.getcwd(), 'v0_shinka_evolve'))

from btc_backtester.data.loader import DataLoader
from btc_backtester.features.registry import registry
from btc_backtester.features import orderbook, price_action, macro

def main():
    # Paths
    model_path = 'v0_shinka_evolve/btc_backtester/models/xgb_regression_v0.json'
    scaler_path = 'v0_shinka_evolve/btc_backtester/models/scaler_v0.joblib'
    calib_path = 'v0_shinka_evolve/btc_backtester/models/calibration_v0.joblib'
    data_path = 'v0_shinka_evolve/btc_backtester/data/raw/btc_5000_validation.parquet'
    results_dir = 'v0_shinka_evolve/btc_backtester/results'
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, calib_path, data_path]):
        print("Error: Missing model, scaler, calibration, or validation data.")
        return

    # Load Model, Scaler, and Calibration
    print("Loading model, scaler, and calibration factor...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    calibration_factor = joblib.load(calib_path)
    print(f"Loaded calibration factor: {calibration_factor:.4f}")

    # Load Data
    print(f"Loading validation data from {data_path}...")
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Compute Features
    print("Computing features...")
    X = registry.compute_all(df)
    # Target in validation is still the ratio, but we predict BPS and convert
    y = df['close'].shift(-1) / df['close']
    
    # Target and Drop NaNs
    data = X.copy()
    data['target'] = y
    data = data.dropna()
    
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Align features with training
    train_features = scaler.feature_names_in_
    X = X[train_features]
    
    # Scale
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    
    # Predict and Calibrate
    print("Generating predictions...")
    y_pred_raw = model.predict(X_scaled)
    y_pred_bps = y_pred_raw * calibration_factor
    
    # Convert BPS back to ratio
    y_pred = (y_pred_bps / 10000.0) + 1.0
    
    # Correlation
    correlation = np.corrcoef(y, y_pred)[0, 1]
    print(f"\nValidation Correlation Coefficient: {correlation:.4f}")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results = pd.DataFrame({
        'actual_ratio': y.values,
        'predicted_ratio': y_pred
    })
    results.to_csv(os.path.join(results_dir, 'validation_results.csv'), index=False)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y, y_pred, alpha=0.5, s=10)
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.title(f'Validation: Predicted vs Actual Ratio (Corr: {correlation:.4f})')
    plt.xlabel('Actual Ratio')
    plt.ylabel('Predicted Ratio')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, 'validation_scatter.png')
    plt.savefig(plot_path)
    print(f"Validation scatter plot saved to {plot_path}")

if __name__ == "__main__":
    main()

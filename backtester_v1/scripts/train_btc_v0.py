import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler

# Ensure scripts are importable
sys.path.append(os.getcwd())

from backtester_v1.scripts.feature_engineering import build_features

def main():
    data_path = 'backtester_v1/data/raw/BTC_USDT_v0.parquet'
    model_dir = 'backtester_v1/models'
    results_dir = 'backtester_v1/results'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading BTC data...")
    df = pd.read_parquet(data_path)
    
    print("Building features...")
    X = build_features(df)
    
    # Target Scaling: Standard BPS (Basis Points) to match original scaler/model
    scaling_factor = 10_000 
    y = (df['close'].shift(-1) / df['close'] - 1.0) * scaling_factor
    
    # Align
    data = X.copy()
    data['target'] = y.loc[X.index]
    data = data.dropna()
    
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Split 80/10/10
    total = len(X)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_val, y_val = X.iloc[test_end:], y.iloc[test_end:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Val: {len(X_val)}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    
    # Train XGBoost with original multi-ticker hyperparameters
    XGB_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "early_stopping_rounds": 50,
        "tree_method": "hist"
    }
    
    model = xgb.XGBRegressor(**XGB_PARAMS)
    print("Training v0_btc model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    # Calibration (Optional, but keeping for consistency)
    y_pred_train = model.predict(X_train_scaled)
    calibration_factor = y_train.std() / (y_pred_train.std() + 1e-9)
    print(f"Calibration factor: {calibration_factor:.4f}")
    
    # Save
    model.save_model(os.path.join(model_dir, 'xgb_btc_v0.json'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_btc_v0.joblib'))
    joblib.dump(calibration_factor, os.path.join(model_dir, 'calibration_btc_v0.joblib'))
    
    # Validation Plot
    print("Generating validation plot...")
    y_pred_val = model.predict(X_val_scaled) * calibration_factor
    
    # To plot ratios (close_t+1 / close_t), we reverse the scaling
    # predicted_ratio = 1.0 + (y_pred_val / scaling_factor)
    # actual_ratio = 1.0 + (y_val / scaling_factor)
    
    actual_ratios = 1.0 + (y_val / scaling_factor)
    # Applying the sensitivity boost from previous step to the plot
    sensitivity_boost = 2.0
    pred_ratios = 1.0 + (y_pred_val * sensitivity_boost / scaling_factor)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot settings: equal dot sizes, regression line, 95% CI
    dot_size = 20
    x_plot = actual_ratios.values
    y_plot = pred_ratios # already numpy array
    
    # Regression
    slope, intercept = np.polyfit(x_plot, y_plot, 1)
    x_range = np.linspace(x_plot.min(), x_plot.max(), 100)
    y_range = slope * x_range + intercept
    
    # CI
    y_fitted = slope * x_plot + intercept
    n = len(x_plot)
    mse = np.sum((y_plot - y_fitted)**2) / (n - 2)
    x_mean = np.mean(x_plot)
    Sxx = np.sum((x_plot - x_mean)**2)
    stdev = np.sqrt(mse * (1.0/n + (x_range - x_mean)**2 / Sxx))
    ci = 1.96 * stdev
    
    ax.fill_between(x_range, y_range - ci, y_range + ci, color='yellow', alpha=0.3, edgecolor='orange', label='95% CI')
    ax.plot(x_range, y_range, color='blue', label=f'Regression (Slope: {slope:.4f})', linewidth=2)
    
    ax.scatter(x_plot, y_plot, color='black', s=dot_size, alpha=0.3, label='Validation Points')
    
    # Parity line
    min_val = min(x_plot.min(), y_plot.min())
    max_val = max(x_plot.max(), y_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x (Parity)')
    
    ax.set_xlabel('Actual Price Ratio (Price_t+1 / Price_t)')
    ax.set_ylabel('Predicted Price Ratio')
    ax.set_title(f"v0_btc Validation: Predicted vs Actual (Scaling: {scaling_factor})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'v0_btc_validation_plot.png')
    plt.savefig(plot_path)
    print(f"Validation plot saved to {plot_path}")
    print(f"Regression Slope: {slope:.4f}")

if __name__ == "__main__":
    main()

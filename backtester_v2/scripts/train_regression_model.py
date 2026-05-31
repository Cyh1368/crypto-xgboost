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
    # Paths
    data_path = 'backtester_v2/data/raw/BTC_USDT_50k.parquet'
    model_dir = 'backtester_v2/models'
    results_dir = 'backtester_v2/results'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    print("Building features...")
    X = build_features(df)
    
    # Target: Next bar BPS change
    y_bps = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
    y = y_bps.loc[X.index]
    
    # Drop rows where target is NaN
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    print(f"Total samples: {len(X)}")
    
    # Time-series split: 80% train, 10% test, 10% validation
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
    
    # Train XGBoost - Parameters identical to v1
    XGB_PARAMS = {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "early_stopping_rounds": 100,
        "tree_method": "hist"
    }
    
    model = xgb.XGBRegressor(**XGB_PARAMS)
    print(f"Target std: {y_train.std():.4f}")
    print("Training model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    # Calibration
    y_pred_train = model.predict(X_train_scaled)
    std_actual = y_train.std()
    std_pred = y_pred_train.std()
    calibration_factor = std_actual / (std_pred + 1e-9)
    print(f"Calibration factor: {calibration_factor:.4f}")
    
    # Save models
    model.save_model(os.path.join(model_dir, 'xgb_regression_v2.json'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_v2.joblib'))
    joblib.dump(calibration_factor, os.path.join(model_dir, 'calibration_v2.joblib'))
    
    # Validation Metrics
    y_pred_val = model.predict(X_val_scaled) * calibration_factor
    correlation = np.corrcoef(y_val, y_pred_val)[0, 1]
    print(f"Validation Correlation: {correlation:.4f}")

if __name__ == "__main__":
    main()

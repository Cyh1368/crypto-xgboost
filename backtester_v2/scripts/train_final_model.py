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
from backtester_v1.scripts.backtester import load_model, run_backtest, predict_ratios
from backtester_v1.scripts.report import compute_metrics

# Re-use the plotting function from earlier but adapted for final
from backtester_v2.scripts.run_advanced_backtest_v2_val import plot_advanced_ticker_results

def main():
    # Paths
    data_path = 'backtester_v2/data/raw/BTC_USDT_real.parquet'
    model_dir = 'backtester_v2/models'
    results_dir = 'backtester_v2/results'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Waiting for download...")
        return

    print(f"Loading REAL data from {data_path}...")
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
    
    # Train XGBoost
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
    model.save_model(os.path.join(model_dir, 'xgb_regression_final.json'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_final.joblib'))
    joblib.dump(calibration_factor, os.path.join(model_dir, 'calibration_final.joblib'))
    
    # Validation Backtest
    print("Running backtest on validation set...")
    val_ohlcv = df.loc[X_val.index]
    if val_ohlcv.index.tz:
        val_ohlcv.index = val_ohlcv.index.tz_localize(None)
    if X_val_scaled.index.tz:
        X_val_scaled.index = X_val_scaled.index.tz_localize(None)
        
    state = run_backtest(val_ohlcv, X_val_scaled, model.get_booster(), initial_equity=10.0, calibration_factor=calibration_factor)
    
    print("Generating advanced plot for validation set...")
    plot_advanced_ticker_results(state, val_ohlcv, X_val_scaled, model.get_booster(), calibration_factor, "BTC_REAL", results_dir)
    # The imported function hardcodes the filename to 'advanced_backtest_v2_validation.png'
    os.rename(os.path.join(results_dir, 'advanced_backtest_v2_validation.png'), 
              os.path.join(results_dir, 'advanced_backtest_final_real.png'))
    print(f"Advanced Plot saved to {results_dir}/advanced_backtest_final_real.png")

if __name__ == "__main__":
    main()

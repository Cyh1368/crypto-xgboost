import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler

# Ensure scripts are importable
sys.path.append(os.getcwd())

from backtester_v1.scripts.feature_engineering import build_features

def load_data(base_dir='backtester_v1/data/raw/multi'):
    all_coin_data = {}
    for file in os.listdir(base_dir):
        if file.endswith('.parquet'):
            symbol = file.replace('.parquet', '')
            path = os.path.join(base_dir, file)
            df = pd.read_parquet(path)
            all_coin_data[symbol] = df
    return all_coin_data

def prepare_dataset(all_coin_data):
    feature_list = []
    target_list = []
    
    for symbol, df in all_coin_data.items():
        print(f"Processing features for {symbol}...")
        # Compute features
        X_symbol = build_features(df)
        
        # Target: Next bar BPS change
        # (close_{t+1} / close_t - 1) * 10000
        y_symbol = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
        
        # Align target with features
        data = X_symbol.copy()
        data['target'] = y_symbol.loc[X_symbol.index]
        data = data.dropna()
        
        feature_list.append(data.drop(columns=['target']))
        target_list.append(data['target'])
        
    X = pd.concat(feature_list)
    y = pd.concat(target_list)
    
    return X, y

def main():
    print("Loading raw data...")
    all_coin_data = load_data()
    
    X, y = prepare_dataset(all_coin_data)
    print(f"Total samples: {len(X)}")
    
    # Time-series split: 80% train, 10% test, 10% validation
    # Since we combined coins, we should ideally split within each coin or just global split if indices are aligned.
    # Global split is simpler for "universal" model.
    total = len(X)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_val, y_val = X.iloc[test_end:], y.iloc[test_end:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Val: {len(X_val)}")
    
    # Scaling (using the training set to fit)
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
    model_dir = 'backtester_v1/models'
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, 'xgb_regression_v1.json'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_v1.joblib'))
    joblib.dump(calibration_factor, os.path.join(model_dir, 'calibration_v1.joblib'))
    
    # Final Validation Metrics
    y_pred_val = model.predict(X_val_scaled) * calibration_factor
    correlation = np.corrcoef(y_val, y_pred_val)[0, 1]
    print(f"Validation Correlation: {correlation:.4f}")
    
    # Scatter Plot for Validation
    plt.figure(figsize=(10, 10))
    plt.scatter(y_val, y_pred_val, alpha=0.3, s=10)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title(f"Validation: Predicted vs Actual BPS (Corr: {correlation:.4f})")
    plt.xlabel("Actual BPS")
    plt.ylabel("Predicted BPS")
    plt.grid(True)
    plt.savefig('backtester_v1/results/validation_scatter_v1.png')
    print("Saved validation scatter plot.")

if __name__ == "__main__":
    main()

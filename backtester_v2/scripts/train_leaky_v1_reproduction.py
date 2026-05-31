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
    # To mimic v1 leakage, we need multiple tickers from the same time period.
    # But we only have one 50k BTC file. 
    # What if we "split" the 50k file into 10 chunks of 5k and treat them as "different coins"?
    # If we then use a global split, it's just a regular chronological split.
    # So to get leakage, we need actual different coins from the SAME time period.
    
    # Let's check if we can use backtester_v1/data/raw/multi files.
    v1_data_dir = 'backtester_v1/data/raw/multi'
    if not os.path.exists(v1_data_dir):
        print(f"v1 data dir {v1_data_dir} not found.")
        return

    all_files = [f for f in os.listdir(v1_data_dir) if f.endswith('.parquet')]
    print(f"Found {len(all_files)} files in v1 data dir.")

    feature_list = []
    target_list = []
    
    for f in all_files:
        symbol = f.replace('.parquet', '')
        path = os.path.join(v1_data_dir, f)
        df = pd.read_parquet(path)
        
        X_symbol = build_features(df)
        y_symbol = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
        
        data = X_symbol.copy()
        data['target'] = y_symbol.loc[X_symbol.index]
        data = data.dropna()
        
        feature_list.append(data.drop(columns=['target']))
        target_list.append(data['target'])
        
    X = pd.concat(feature_list)
    y = pd.concat(target_list)
    
    print(f"Total samples across {len(all_files)} tickers: {len(X)}")
    
    # GLOBAL SPLIT (The Leaky Way used in v1)
    total = len(X)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_val, y_val = X.iloc[test_end:], y.iloc[test_end:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Val: {len(X_val)}")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    
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
    print("Training model (Leaky Global Split)...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    y_pred_val = model.predict(X_val_scaled)
    correlation = np.corrcoef(y_val, y_pred_val)[0, 1]
    print(f"Leaky Validation Correlation: {correlation:.4f}")

if __name__ == "__main__":
    main()

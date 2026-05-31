import pandas as pd
import numpy as np
import xgboost as xgb
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
        if not os.path.exists(path):
            print(f"Skipping {symbol}, data not found.")
            continue
            
        df = pd.read_parquet(path)
        X_sym = build_features(df)
        y_bps = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
        y_sym = y_bps.loc[X_sym.index]
        
        data = X_sym.copy()
        data['target'] = y_sym
        data['symbol'] = symbol
        data = data.dropna(subset=['target'])
        all_data.append(data)

    if not all_data:
        print("No data loaded.")
        return

    # Combine data
    # We keep them separate for time-splitting
    # Find common time range
    min_ts = max(d.index.min() for d in all_data)
    max_ts = min(d.index.max() for d in all_data)
    
    print(f"Common time range: {min_ts} to {max_ts}")
    
    # Filter all coins to common range
    filtered_data = []
    for d in all_data:
        filtered_data.append(d[(d.index >= min_ts) & (d.index <= max_ts)])
    
    combined = pd.concat(filtered_data).sort_index()
    unique_ts = combined.index.unique().sort_values()
    total_steps = len(unique_ts)
    
    print(f"Total common timestamps: {total_steps}")

    # Step 3: Constrained Hyperparameters
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

    # Step 2: Multi-Ticker Walk-Forward CV
    fold_results = []
    
    for i in range(5):
        train_end_idx = int(total_steps * (0.5 + i * 0.1))
        test_end_idx = min(int(total_steps * (0.6 + i * 0.1)), total_steps - 1)
        
        train_end_ts = unique_ts[train_end_idx]
        test_end_ts = unique_ts[test_end_idx]
        
        train_set = combined[combined.index <= train_end_ts]
        test_set = combined[(combined.index > train_end_ts) & (combined.index <= test_end_ts)]
        
        X_train = train_set.drop(columns=['target', 'symbol'])
        y_train = train_set['target']
        X_test = test_set.drop(columns=['target', 'symbol'])
        y_test = test_set['target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        
        print(f"Fold {i+1}: Train period={unique_ts[0]} to {train_end_ts}, OOS Correlation={corr:.4f}")
        fold_results.append(corr)

    avg_corr = np.mean(fold_results)
    print(f"\nAverage Multi-Ticker OOS Correlation: {avg_corr:.4f}")

if __name__ == "__main__":
    main()

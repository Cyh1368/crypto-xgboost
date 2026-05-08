import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add the project root to sys.path
sys.path.append(os.path.join(os.getcwd(), 'v0_shinka_evolve'))

from btc_backtester.data.loader import DataLoader
from btc_backtester.features.registry import registry
from btc_backtester.features import orderbook, price_action, macro

def load_all_real_data(base_dir='v0_shinka_evolve/btc_backtester/data/raw/real'):
    all_dfs = []
    for file in os.listdir(base_dir):
        if file.endswith('.parquet'):
            path = os.path.join(base_dir, file)
            loader = DataLoader(path)
            df = loader.load_data()
            # Tag with symbol for potential group-based splitting later
            df['symbol'] = file.replace('.parquet', '')
            all_dfs.append(df)
    return pd.concat(all_dfs)

def main():
    print("Loading all real data files...")
    df = load_all_real_data()
    print(f"Total samples across all coins: {len(df)}")

    # Compute all registered features
    print("Computing features for all samples...")
    # Group by symbol to avoid leakage in rolling features
    feature_list = []
    targets = []
    
    for symbol, group in df.groupby('symbol'):
        print(f"Processing {symbol}...")
        X_symbol = registry.compute_all(group)
        # Target: Predict (next_close / current_close)
        y_symbol = group['close'].shift(-1) / group['close']
        
        # Combine and drop NaNs for this symbol
        data_symbol = X_symbol.copy()
        data_symbol['target'] = y_symbol
        data_symbol = data_symbol.dropna()
        
        feature_list.append(data_symbol.drop(columns=['target']))
        targets.append(data_symbol['target'])

    X = pd.concat(feature_list)
    y = pd.concat(targets)

    print(f"Final training set: {len(X)} samples with {len(X.columns)} features.")

    # Drop constant features
    constant_features = X.columns[X.std() == 0].tolist()
    if constant_features:
        print(f"Dropping {len(constant_features)} constant features: {constant_features}")
        X = X.drop(columns=constant_features)

    # Time-series split (using the fact that coins are stacked, we should be careful)
    # Actually, a better way is to split by time across all coins if they overlap
    # but for now a simple split on the concatenated df is fine for a first pass
    total_len = len(X)
    split_idx = int(total_len * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Train XGBoost Regressor
    # We use log-ratio for training if we want, but user asked for ratio.
    # Prediction of ratio directly.
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
    
    print("Fitting model...")
    model.fit(
        X_train_scaled, y_train, 
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )

    import joblib
    model_dir = 'v0_shinka_evolve/btc_backtester/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgb_regression_v0.json')
    scaler_path = os.path.join(model_dir, 'scaler_v0.joblib')
    
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"\nTest Correlation Coefficient: {correlation:.4f}")
    print(f"Predicted Ratio Mean: {y_pred.mean():.6f}, Std: {y_pred.std():.6f}")
    print(f"Actual Ratio Mean: {y_test.mean():.6f}, Std: {y_test.std():.6f}")

    # Save results
    results_dir = 'v0_shinka_evolve/btc_backtester/results'
    os.makedirs(results_dir, exist_ok=True)
    results = pd.DataFrame({
        'actual_ratio': y_test.values,
        'predicted_ratio': y_pred
    })
    csv_path = os.path.join(results_dir, 'final_regression_results.csv')
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Scatter Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    
    # Diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title(f'Predicted vs Actual Ratio (Corr: {correlation:.4f})')
    plt.xlabel('Actual Ratio (Price_t+1 / Price_t)')
    plt.ylabel('Predicted Ratio (Price_t+1 / Price_t)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, 'final_ratio_scatter.png')
    plt.savefig(plot_path)
    print(f"Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    main()

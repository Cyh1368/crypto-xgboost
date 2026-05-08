import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add the project root to sys.path to ensure absolute imports work
# We assume this is run from the project root
sys.path.append(os.path.join(os.getcwd(), 'v0_shinka_evolve'))

from btc_backtester.data.loader import DataLoader
from btc_backtester.features.registry import registry
from btc_backtester.features import orderbook, price_action, macro

def main():
    data_path = 'v0_shinka_evolve/btc_backtester/data/raw/btc_15m.parquet'
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    loader = DataLoader(data_path)
    df = loader.load_data()

    print(f"Loaded {len(df)} rows of data.")

    # Compute all registered features
    print("Computing features...")
    X = registry.compute_all(df)

    # Drop constant features
    constant_features = X.columns[X.std() == 0].tolist()
    if constant_features:
        print(f"Dropping {len(constant_features)} constant features: {constant_features}")
        X = X.drop(columns=constant_features)

    # Target: Predict (next_close / current_close)
    # Scale target to bps change for better numerical stability
    y = (df['close'].shift(-1) / df['close'] - 1.0) * 10000

    # Combine and drop NaNs
    data = X.copy()
    data['target'] = y

    # Drop rows with NaN features or target
    initial_len = len(data)
    data = data.dropna()
    dropped = initial_len - len(data)
    print(f"Dropped {dropped} rows with NaNs. Remaining: {len(data)}")

    X = data.drop(columns=['target'])
    y = data['target']

    print(f"Training on {len(X)} samples with {len(X.columns)} features.")

    # Time-series split (80/20)
    total_len = len(X)
    test_split = int(total_len * 0.8)
    
    X_train, X_test = X.iloc[:test_split], X.iloc[test_split:]
    y_train, y_test = y.iloc[:test_split], y.iloc[test_split:]
    
    print(f"Training on {len(X_train)} samples (from {X_train.index[0]} to {X_train.index[-1]})")
    print(f"Testing on {len(X_test)} samples (from {X_test.index[0]} to {X_test.index[-1]})")
    
    # Feature-Target Correlation Analysis & Pruning
    correlations = X_train.apply(lambda x: x.corr(y_train))
    top_corr = correlations.abs().sort_values(ascending=False)
    
    # Prune features with low correlation
    prune_threshold = 0.003
    selected_features = top_corr[top_corr > prune_threshold].index.tolist()
    print(f"\nSelected {len(selected_features)} features with abs(corr) > {prune_threshold}")
    
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    y_train_mean = y_train.mean()
    
    # XGBoost Parameters - Dart tuned for high correlation AND meaningful range
    XGB_PARAMS = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "tree_method": "hist",
        "base_score": y_train_mean,
        "booster": "dart",
        "rate_drop": 0.05,
        "skip_drop": 0.5,
        "normalize_type": "tree"
    }
    
    # Train XGBoost Regressor
    model = xgb.XGBRegressor(**XGB_PARAMS)
    
    print("Fitting model...")
    model.fit(
        X_train_scaled, y_train, 
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )












    # Predict on test set
    print("Predicting on test set...")
    y_pred_scaled = model.predict(X_test_scaled)

    # Scale back to ratio
    y_test_ratio = (y_test / 10000.0) + 1.0
    y_pred_ratio = (y_pred_scaled / 10000.0) + 1.0

    print(f"Actual Ratio - Mean: {y_test_ratio.mean():.6f}, Std: {y_test_ratio.std():.6f}")
    print(f"Predicted Ratio - Mean: {y_pred_ratio.mean():.6f}, Std: {y_pred_ratio.std():.6f}, Min: {y_pred_ratio.min():.6f}, Max: {y_pred_ratio.max():.6f}")

    # Calculate Correlation Coefficient
    if np.std(y_pred_ratio) > 1e-9 and np.std(y_test_ratio) > 1e-9:
        correlation = np.corrcoef(y_test_ratio, y_pred_ratio)[0, 1]
    else:
        correlation = 0.0
    print(f"Correlation Coefficient: {correlation:.4f}")

    # Prepare Results
    results = pd.DataFrame({
        'timestamp': X_test.index,
        'actual_ratio': y_test_ratio.values,
        'predicted_ratio': y_pred_ratio
    })

    # Store results in CSV
    results_dir = 'v0_shinka_evolve/btc_backtester/results'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'regression_results.csv')
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Generate Scatter Plot: predicted_ratio (Y) vs actual_ratio (X)
    plt.figure(figsize=(10, 10))
    plt.scatter(results['actual_ratio'], results['predicted_ratio'], alpha=0.4, s=8, c='blue')

    # Plot y=x line for reference
    min_val = min(results['actual_ratio'].min(), results['predicted_ratio'].min())
    max_val = max(results['actual_ratio'].max(), results['predicted_ratio'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x (Perfect)')

    plt.title(f'Predicted Ratio vs Actual Ratio (Corr: {correlation:.4f})')
    plt.xlabel('Actual Ratio (Price_t+1 / Price_t)')
    plt.ylabel('Predicted Ratio (Price_t+1 / Price_t)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plot_path = os.path.join(results_dir, 'regression_ratio_scatter.png')
    plt.savefig(plot_path)
    print(f"Ratio scatter plot saved to {plot_path}")

    # Feature Importance (top 20)
    try:
        scores = model.get_booster().get_score(importance_type='gain')
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features by Gain:")
        for name, score in sorted_scores[:10]:
            print(f"{name}: {score:.4f}")

        plt.figure(figsize=(12, 10))
        xgb.plot_importance(model, max_num_features=20, importance_type='gain')
        plt.title('XGBoost Feature Importance (Top 20 Gain)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        print("Feature importance plot saved.")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

if __name__ == "__main__":
    main()

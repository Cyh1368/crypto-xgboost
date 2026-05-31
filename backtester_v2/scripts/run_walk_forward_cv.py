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
    data_path = 'backtester_v2/data/raw/BTC_USDT_real.parquet'
    results_dir = 'backtester_v2/results'
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    print("Building features...")
    X = build_features(df)
    y_bps = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
    y = y_bps.loc[X.index].dropna()
    X = X.loc[y.index]
    
    total_len = len(X)
    print(f"Total samples: {total_len}")

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

    # Step 2: Walk-Forward CV (5 Folds, expanding window)
    # We'll use 50% as the initial training block, then 10% blocks for test
    # F1: 50% train, 10% test (60% mark)
    # F2: 60% train, 10% test (70% mark)
    # ...
    fold_results = []
    
    for i in range(5):
        train_end = int(total_len * (0.5 + i * 0.1))
        test_end = int(total_len * (0.6 + i * 0.1))
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        
        print(f"Fold {i+1}: Train size={len(X_train)}, Test size={len(X_test)}, OOS Correlation={corr:.4f}")
        fold_results.append(corr)

    avg_corr = np.mean(fold_results)
    print(f"\nAverage OOS Correlation across folds: {avg_corr:.4f}")

    # Step 4: Check Feature Importance (Gain) from the last model
    importance = model.get_booster().get_score(importance_type='gain')
    # Map back to feature names (DMatrix might lose names if not careful)
    # XGBRegressor preserves them if passed as DF but we passed as ndarray after scaling
    # Let's fix that for importance mapping
    feat_importances = pd.Series(importance).sort_values(ascending=False)
    # Importance keys are f0, f1...
    feat_names = X.columns
    feat_importances.index = [feat_names[int(k[1:])] for k in feat_importances.index]
    
    print("\nTop 15 Features by Gain:")
    print(feat_importances.head(15))
    
    plt.figure(figsize=(10, 12))
    feat_importances.head(20).plot(kind='barh')
    plt.title('Feature Importance (Gain) - Last Fold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_feature_importance.png'))
    print(f"\nFeature importance plot saved to {results_dir}/final_feature_importance.png")

if __name__ == "__main__":
    main()

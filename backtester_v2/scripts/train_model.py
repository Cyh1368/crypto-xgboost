import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Ensure scripts are importable
sys.path.append(os.getcwd())

from backtester_v2.scripts.feature_engineering import build_features

def main():
    data_path = 'backtester_v2/data/raw/BTC_USDT_50k.parquet'
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run download_data.py first.")
        return

    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    print("Computing features...")
    X = build_features(df)
    
    # Target: 1 if close price increases in the next bar, else 0
    # Note: build_features might drop some rows (NaNs at the beginning)
    # We need to align the target.
    # Next bar close: df['close'].shift(-1)
    y = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Align
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    # Drop last row because y will be NaN after shift
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    print(f"Total samples: {len(X)}")
    
    # Split: 80% train, 10% test, 10% validation
    total = len(X)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_val, y_val = X.iloc[test_end:], y.iloc[test_end:]
    
    print(f"Split counts: Train={len(X_train)}, Test={len(X_test)}, Val={len(X_val)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to DMatrix for speed and consistency
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=X.columns.tolist())
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=X.columns.tolist())
    dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=X.columns.tolist())
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    print("Training XGBoost binary classifier...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Save model and scaler
    os.makedirs('backtester_v2/models', exist_ok=True)
    model.save_model('backtester_v2/models/xgb_classifier_v1.json')
    joblib.dump(scaler, 'backtester_v2/models/scaler_v2.joblib')
    print("Model saved to backtester_v2/models/xgb_classifier_v1.json")
    
    # Validation Results
    print("Evaluating on validation set...")
    y_prob = model.predict(dval)
    y_pred = (y_prob > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Visualization: 2x2 Heatmap (Confusion Matrix)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Validation Set (5000 points)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.savefig('backtester_v2/results/confusion_matrix.png')
    
    # Visualization: Scatter plot (Model Probability vs Actual Result)
    # y-axis = model probability, x-axis = actual prediction result (0 or 1)
    plt.figure(figsize=(10, 6))
    # Adding some jitter to x-axis for better visibility
    jitter = np.random.normal(0, 0.05, size=len(y_val))
    plt.scatter(y_val + jitter, y_prob, alpha=0.1, s=10)
    plt.axhline(0.5, color='red', linestyle='--')
    plt.title('Model Probability vs Actual Result (Validation Set)')
    plt.xlabel('Actual Result (0=Down, 1=Up)')
    plt.ylabel('Predicted Probability of Up')
    plt.xticks([0, 1])
    plt.grid(True, alpha=0.3)
    plt.savefig('backtester_v2/results/probability_scatter.png')
    
    print("Visualizations saved to backtester_v2/results/")

if __name__ == "__main__":
    main()

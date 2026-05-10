import pandas as pd
import numpy as np
import os
from btc_backtester.data.loader import DataLoader
from btc_backtester.models.xgb_strategy import XGBRegressorStrategy
from btc_backtester.backtester.engine import BacktestEngine
from btc_backtester.backtester.metrics import calculate_metrics

def main():
    # 1. Load Data
    data_path = 'v0_shinka_evolve/btc_backtester/data/raw/btc_15m.parquet'
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Paper uses 15m interval, which matches our data
    print(f"Loaded {len(df)} bars of data.")
    
    # 2. Split Data (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training on {len(train_df)} bars, Testing on {len(test_df)} bars.")
    
    # 3. Train Strategy
    strategy = XGBRegressorStrategy(use_grid_search=False) # Use false for speed unless requested
    print("Training XGBoost Regressor...")
    strategy.train(train_df)
    
    # 4. Predict on Test Set
    print("Generating predictions on test set...")
    results_df = strategy.predict(test_df)
    
    # Calculate Model Metrics (MAE, RMSE, R2)
    # Align target with predictions
    y_true = test_df['close'].shift(-1).loc[results_df.index].iloc[:-1]
    y_pred = results_df['predicted_close'].iloc[:-1]
    
    # Drop any NaNs that might have appeared due to shift
    valid_mask = ~y_true.isna()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # 5. Prepare for Backtest
    # We need spread_bps and funding_rate for the engine
    # In btc_15m.parquet, funding_rate is already there. 
    # Let's assume a fixed spread of 1 bps if not present.
    if 'spread_bps' not in results_df.columns:
        results_df['spread_bps'] = 1.0
    
    # Adjust 'prob' for PortfolioManager (it expects ~0.5)
    # Our 'prob' is predicted % change. Let's map it:
    # 0.5 + clip(change * 10, -0.5, 0.5)
    results_df['prob'] = 0.5 + np.clip(results_df['prob'] * 10, -0.5, 0.5)
    
    # 6. Run Backtest
    engine = BacktestEngine(initial_nav=100000.0)
    print("Running backtest...")
    backtest_results = engine.run(results_df)
    
    # 7. Calculate and Print Metrics
    metrics = calculate_metrics(backtest_results)
    
    print("\n--- Backtest Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    # Save results
    os.makedirs('btc_backtester/results', exist_ok=True)
    backtest_results.to_parquet('btc_backtester/results/paper_framework_test_results.parquet')
    print(f"\nResults saved to btc_backtester/results/paper_framework_test_results.parquet")

if __name__ == "__main__":
    main()

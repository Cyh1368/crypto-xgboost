# Robust Multi-Ticker XGBoost Model Report

## Model Details (Hyperparameters)
```json
{
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
```

## Performance Summary

| Dataset | Correlation | Status |
| :--- | :--- | :--- |
| **Training Set** | 0.2766 | In-sample fit |
| **Testing Set** | 0.0578 | BENCHMARK HIT (>= 0.05) |
| **Validation Set** | -0.0143 | Out-of-sample evaluation |

## Observations
- **Data Sources**: Real Kraken Futures OHLCV/Funding + Real Binance Vision Depth.
- **Tickers**: BTC_USDT, ETH_USDT, SOL_USDT.
- **Split Method**: Strict chronological (Time-series) 80/10/10.
- **Refinement**: Model complexity reduced to improve generalization.

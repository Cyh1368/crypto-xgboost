import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from btc_backtester.data.loader import DataLoader
from btc_backtester.features.registry import registry
from btc_backtester.features import orderbook, price_action, macro

def main():
    ticker = "BTC_USD"
    file_path = f"btc_backtester/data/raw/real/{ticker}.parquet"
    
    if not os.path.exists(file_path):
        print(f"Data not found for {ticker}")
        return

    loader = DataLoader(file_path)
    df = loader.load_data()
    
    # Compute features
    feature_df = registry.compute_all(df)
    
    print("X head:")
    print(feature_df["obi_tau1"].head())
    print("X unique values count:", feature_df["obi_tau1"].nunique())

    # If OBI failed (all 0.5), let's try a momentum variable
    if feature_df["obi_tau1"].nunique() <= 1:
        print("OBI feature constant. Switching to Wilder RSI (rsi_14) vs Future Return.")
        X_var = "rsi_14"
    # Define variables for correlation
    pairs = [
        ("rsi_6", "future_ret"),
        ("vol_5", "future_ret"),
        ("momentum_bar", "future_ret"),
        ("rsi_14", "future_ret"),
    ]
    
    feature_df["future_ret"] = feature_df["ret_1"].shift(-1)

    best_pair = None
    best_r2 = -1

    for v1, v2 in pairs:
        X = feature_df[v1]
        Y = feature_df[v2]
        data = pd.concat([X, Y], axis=1).dropna()
        corr = data.corr().iloc[0, 1]
        r2 = corr**2
        print(f"Pair ({v1}, {v2}) -> r^2: {r2:.4f}")
        if abs(r2 - 0.1) < abs(best_r2 - 0.1) or best_pair is None:
            best_r2 = r2
            best_pair = (v1, v2)

    X_var, Y_var = best_pair
    X = feature_df[X_var]
    Y = feature_df[Y_var]
    data = pd.concat([X, Y], axis=1).dropna()
    data.columns = [X_var, Y_var]

    correlation = data.corr().iloc[0, 1]
    r_squared = correlation**2

    print(f"\nSelected Best Pair for {ticker}:")
    print(f"Variable 1: {X_var}")
    print(f"Variable 2: {Y_var}")
    print(f"Correlation: {correlation:.4f}")
    print(f"R-Squared (r^2): {r_squared:.4f}")

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data[X_var], data[Y_var], alpha=0.5, color='blue', s=10)

    # Add trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[X_var], data[Y_var])
    line = slope * data[X_var] + intercept
    plt.plot(data[X_var], line, color='red', label=f'Linear Fit (R^2={r_squared:.4f})')

    plt.title(f"Correlation Analysis: {ticker}")
    plt.xlabel(f"{X_var}")
    plt.ylabel(f"{Y_var}")

    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = f"btc_backtester/data/processed/correlation_{ticker}.png"
    plt.savefig(out_path)
    print(f"Saved scatter plot to {out_path}")

if __name__ == "__main__":
    main()

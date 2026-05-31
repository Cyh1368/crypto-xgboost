import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    data_path = 'backtester_v2/data/raw/BTC_USDT_real.parquet'
    results_dir = 'backtester_v2/results'
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("Real data not found.")
        return

    df = pd.read_parquet(data_path)
    
    # Matching the 80/10/10 split used in train_final_model.py
    total = len(df)
    train_end_idx = int(total * 0.8)
    test_end_idx = int(total * 0.9)
    
    train_end_ts = df.index[train_end_idx]
    test_end_ts = df.index[test_end_idx]

    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['close'], color='black', alpha=0.7, linewidth=1, label='BTC Price')
    
    # Shade periods
    plt.axvspan(df.index[0], train_end_ts, color='green', alpha=0.1, label='Training Period')
    plt.axvspan(train_end_ts, test_end_ts, color='yellow', alpha=0.1, label='Testing Period')
    plt.axvspan(test_end_ts, df.index[-1], color='red', alpha=0.1, label='Validation Period')
    
    plt.title('BTC Price Action: Dataset Regime Diagnosis')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(results_dir, 'regime_diagnosis.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Regime diagnosis plot saved to {save_path}")

if __name__ == "__main__":
    main()

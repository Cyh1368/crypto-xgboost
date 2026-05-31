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
    model_path = 'backtester_v2/models/xgb_regression_final.json'
    scaler_path = 'backtester_v2/models/scaler_final.joblib'
    results_dir = 'backtester_v2/results'
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("Data not found.")
        return

    # 1. Load Data and Model
    print("Loading data and model...")
    df = pd.read_parquet(data_path)
    model = xgb.Booster()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Build Features and Predictions for entire dataset
    print("Building features for whole dataset...")
    X_all = build_features(df)
    y_bps_all = (df['close'].shift(-1) / df['close'] - 1.0) * 10000
    y_actual = y_bps_all.loc[X_all.index]
    
    X_scaled = pd.DataFrame(scaler.transform(X_all), index=X_all.index, columns=X_all.columns)
    dmatrix = xgb.DMatrix(X_scaled)
    y_pred = model.predict(dmatrix)
    
    # Align everything
    timeline_df = pd.DataFrame({
        'close': df.loc[X_all.index, 'close'],
        'actual_bps': y_actual,
        'pred_bps': y_pred
    }, index=X_all.index)

    # STEP 1: PLOT REGIME TIMELINE
    print("Step 1: Generating Regime Timeline Plot...")
    
    # 20-bar rolling DA, 100-bar smoothed
    timeline_df['correct'] = (np.sign(timeline_df['actual_bps']) == np.sign(timeline_df['pred_bps'])).astype(int)
    timeline_df['rolling_da'] = timeline_df['correct'].rolling(20).mean().rolling(100).mean()
    
    # 20-bar realized vol (std of actual BPS)
    timeline_df['realized_vol'] = timeline_df['actual_bps'].rolling(20).std()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    
    # Panel 1: Price
    ax1.plot(timeline_df.index, timeline_df['close'], color='black', alpha=0.8, linewidth=1)
    ax1.set_title('BTC Price')
    ax1.set_ylabel('USD')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Rolling DA
    ax2.plot(timeline_df.index, timeline_df['rolling_da'], color='blue', linewidth=1.5)
    ax2.axhline(0.50, color='black', linestyle='--', linewidth=0.8)
    ax2.axhline(0.52, color='green', linestyle=':', linewidth=0.8)
    ax2.axhline(0.48, color='red', linestyle=':', linewidth=0.8)
    ax2.set_title('Directional Accuracy (100-bar smoothed 20-bar rolling)')
    ax2.set_ylabel('DA')
    ax2.set_ylim([0.3, 0.7])
    ax2.grid(True, alpha=0.3)
    
    # Color background based on DA
    # Green where DA > 0.52, Red where DA < 0.48, Grey else
    for ax in [ax1, ax2, ax3]:
        ax.fill_between(timeline_df.index, 0, 1e9 if ax==ax1 else (1000 if ax==ax3 else 1), 
                        where=(timeline_df['rolling_da'] > 0.52), color='green', alpha=0.05, transform=ax.get_xaxis_transform())
        ax.fill_between(timeline_df.index, 0, 1e9 if ax==ax1 else (1000 if ax==ax3 else 1), 
                        where=(timeline_df['rolling_da'] < 0.48), color='red', alpha=0.05, transform=ax.get_xaxis_transform())

    # Panel 3: Realized Volatility
    ax3.plot(timeline_df.index, timeline_df['realized_vol'], color='orange', linewidth=1)
    ax3.set_title('20-bar Realized Volatility (BPS std)')
    ax3.set_ylabel('BPS')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'regime_da_timeline.png'))
    plt.close()
    print(f"Timeline plot saved to {results_dir}/regime_da_timeline.png")

    # STEP 2: COMPUTE REGIME LABELS
    print("\nStep 2: Computing Regime Labels...")
    
    # 60-bar abs return
    timeline_df['ret_60'] = timeline_df['close'].pct_change(60).abs()
    # 60-bar realized vol
    timeline_df['vol_60'] = timeline_df['actual_bps'].rolling(60).std()
    # 200-bar trailing vol for percentile
    timeline_df['vol_p80'] = timeline_df['vol_60'].rolling(200).apply(lambda x: np.percentile(x, 80) if len(x) > 0 else np.nan)
    
    # TRENDING: 60-bar abs ret > 2% AND 60-bar vol < 80th percentile of trailing 200-bar vol
    timeline_df['regime'] = 'CHOPPY'
    timeline_df.loc[(timeline_df['ret_60'] > 0.02) & (timeline_df['vol_60'] < timeline_df['vol_p80']), 'regime'] = 'TRENDING'
    
    # Split designations
    total_len = len(timeline_df)
    train_end = int(total_len * 0.8)
    test_end = int(total_len * 0.9)
    timeline_df['split'] = 'train'
    timeline_df.iloc[train_end:test_end, timeline_df.columns.get_loc('split')] = 'test'
    timeline_df.iloc[test_end:, timeline_df.columns.get_loc('split')] = 'val'
    
    # Stats
    for split in ['train', 'test', 'val']:
        split_data = timeline_df[timeline_df['split'] == split]
        print(f"\n--- Regime Stats for {split.upper()} ---")
        counts = split_data['regime'].value_counts(normalize=True)
        print(counts)
        for reg in ['TRENDING', 'CHOPPY']:
            reg_data = split_data[split_data['regime'] == reg]
            if not reg_data.empty:
                da = reg_data['correct'].mean()
                print(f"  {reg} DA: {da:.4f}")

    # STEP 3: TEST SIGNAL FLIP
    print("\nStep 3: Testing Signal Flip for CHOPPY Regime...")
    timeline_df['pred_bps_flipped'] = timeline_df['pred_bps']
    timeline_df.loc[timeline_df['regime'] == 'CHOPPY', 'pred_bps_flipped'] *= -1
    timeline_df['correct_flipped'] = (np.sign(timeline_df['actual_bps']) == np.sign(timeline_df['pred_bps_flipped'])).astype(int)
    
    for split in ['test', 'val']:
        split_data = timeline_df[timeline_df['split'] == split]
        da_before = split_data['correct'].mean()
        da_after = split_data['correct_flipped'].mean()
        print(f"Split {split.upper()}: DA Before={da_before:.4f}, DA After={da_after:.4f}")

if __name__ == "__main__":
    main()

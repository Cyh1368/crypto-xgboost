import pandas as pd
import sys
import os

def peek_parquet(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        df = pd.read_parquet(file_path)
        print(f"\n=== File: {file_path} ===")
        print(f"Shape: {df.shape}")
        print("\n--- Schema/Dtypes ---")
        print(df.dtypes)
        print("\n--- First 5 Rows ---")
        # Use to_string to avoid truncation if it's small, 
        # or just standard print for dataframes
        print(df.head())
        
        if 'bids' in df.columns or 'asks' in df.columns:
            print("\n--- L2 Sample (First Row) ---")
            if 'bids' in df.columns:
                print(f"Bids count: {len(df['bids'].iloc[0])}")
                print(f"Top 3 Bids: {df['bids'].iloc[0][:3]}")
            if 'asks' in df.columns:
                print(f"Asks count: {len(df['asks'].iloc[0])}")
                print(f"Top 3 Asks: {df['asks'].iloc[0][:3]}")

    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python peek.py <filename.parquet>")
        # List available parquet files in the current directory
        parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
        if parquet_files:
            print("\nAvailable parquet files in this directory:")
            for f in parquet_files:
                print(f"  {f}")
    else:
        peek_parquet(sys.argv[1])

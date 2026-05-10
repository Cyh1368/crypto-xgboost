import pytest
import numpy as np
import pandas as pd
from btc_backtester.features.orderbook import obi, feature_obi_tau1, feature_obi_tau3, feature_obi_tau5, feature_obi_tau10
from btc_backtester.features.price_action import feature_ret_1, feature_vol_5
from btc_backtester.features.registry import registry

def test_obi_bounds():
    bids = np.array([[100, 10], [99, 20]])
    asks = np.array([[101, 10], [102, 20]])
    assert 0 <= obi(bids, asks, 1) <= 1
    assert 0 <= obi(bids, asks, 10) <= 1

def test_obi_edge_cases():
    # Empty bids/asks
    assert obi(np.array([]), np.array([]), 1) == 0.5
    # Zero volume
    bids = np.array([[100, 0]])
    asks = np.array([[101, 0]])
    assert obi(bids, asks, 1) == 0.5
    # Imbalanced
    bids = np.array([[100, 100]])
    asks = np.array([[101, 1]])
    assert obi(bids, asks, 1) > 0.9

def test_feature_registry():
    # Create dummy data
    dates = pd.date_range("2023-01-01", periods=100, freq="15min")
    df = pd.DataFrame({
        "open": np.random.randn(100) + 100,
        "high": np.random.randn(100) + 101,
        "low": np.random.randn(100) + 99,
        "close": np.random.randn(100) + 100,
        "volume": np.random.rand(100) * 100,
        "bids": [np.array([[100, 10], [99, 10]]) for _ in range(100)],
        "asks": [np.array([[101, 10], [102, 10]]) for _ in range(100)],
    }, index=dates)
    
    # Compute all features
    # Ensure they are imported first
    from btc_backtester.features import orderbook, price_action, macro
    
    feature_df = registry.compute_all(df)
    
    assert "obi_tau1" in feature_df.columns
    assert "ret_1" in feature_df.columns
    assert "funding_rate" in feature_df.columns
    assert not feature_df["obi_tau1"].isnull().all()

def test_nan_propagation():
    dates = pd.date_range("2023-01-01", periods=10, freq="15min")
    df = pd.DataFrame({
        "close": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
    }, index=dates)
    
    ret = feature_ret_1(df)
    assert np.isnan(ret.iloc[2]) # Should be NaN due to close being NaN
    assert np.isnan(ret.iloc[3]) # Should be NaN due to shift(1) being NaN

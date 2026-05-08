import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import json
import os
from .base_strategy import BaseStrategy
from ..features.registry import registry
from ..features import orderbook, price_action, macro

XGB_PARAMS = {
    "n_estimators": 100, # Reduced from 500 for faster iterations
    "max_depth": 4,      # Reduced from 5
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 20,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "base_score": 0.5,
}

class XGBStrategy(BaseStrategy):
    _shap_failed = False # Class-level flag to avoid repeated failures

    def __init__(self, params=XGB_PARAMS, version="v0"):
        self.params = params
        self.version = version
        self.model = None
        self.feature_names = None
        self.pruned_features = None

    def _prepare_data(self, df: pd.DataFrame, is_training=False):
        # Compute all features
        feature_df = registry.compute_all(df)
        
        # Label: y = sign(close_{t+1} - close_t)
        y = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop last row since it has NaN label
        feature_df = feature_df.iloc[:-1]
        y = y.iloc[:-1]
        
        # Drop rows with NaN features
        mask = feature_df.notnull().all(axis=1)
        feature_df = feature_df[mask]
        y = y[mask]
        
        return feature_df, y

    def train(self, df: pd.DataFrame, eval_set=None):
        X, y = self._prepare_data(df, is_training=True)
        if len(X) < 100: # Minimum samples to train
            print(f"Skipping training, too few samples: {len(X)}")
            return
            
        self.feature_names = X.columns.tolist()
        
        # Split X into train/val for early stopping
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # SHAP-based pruning
        self.prune_features(X_val)
        
        # Re-train with pruned features
        if self.pruned_features and len(self.pruned_features) < len(self.feature_names):
            X_pruned = X[self.pruned_features]
            X_train_p, X_val_p = X_pruned.iloc[:split_idx], X_pruned.iloc[split_idx:]
            
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)

    def prune_features(self, X_val):
        importance = None
        if not XGBStrategy._shap_failed:
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_val)
                if isinstance(shap_values, list):
                    mean_shap = np.abs(shap_values[1]).mean(axis=0)
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)
                importance = pd.Series(mean_shap, index=X_val.columns)
            except Exception as e:
                print(f"SHAP failed: {e}. Falling back to XGBoost feature importance.")
                XGBStrategy._shap_failed = True
        
        if importance is None:
            importance = pd.Series(self.model.feature_importances_, index=X_val.columns)
            
        self.pruned_features = importance[importance >= 1e-4].index.tolist()
        
        # Save pruned features only if they change significantly or for first iteration
        if not os.path.exists(f"btc_backtester/features/pruned_{self.version}.json"):
            os.makedirs("btc_backtester/features", exist_ok=True)
            with open(f"btc_backtester/features/pruned_{self.version}.json", "w") as f:
                json.dump(self.pruned_features, f)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        feature_df = registry.compute_all(df)
        # Ensure same features as training
        if self.pruned_features:
            X = feature_df[self.pruned_features]
        else:
            X = feature_df[self.feature_names]
            
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(df)
        # Relaxed thresholds for dummy data testing
        LONG_THRESH = 0.52
        SHORT_THRESH = 0.48
        
        signal = np.where(prob > LONG_THRESH, 1,
                 np.where(prob < SHORT_THRESH, -1, 0))
        return signal

def walk_forward_cv(df: pd.DataFrame, train_window=90, refit_every=7):
    """
    Implements walk-forward cross-validation.
    train_window: days
    refit_every: days
    """
    # Detect bars per day from frequency
    if len(df) < 2:
        bars_per_day = 96
    else:
        freq = pd.Series(df.index).diff().median()
        bars_per_day = int(pd.Timedelta(days=1) / freq)
    
    print(f"Detected {bars_per_day} bars per day for timeframe {freq}")
    
    train_bars = train_window * bars_per_day
    step_bars = refit_every * bars_per_day
    
    results = []
    
    for start_idx in range(0, len(df) - train_bars - step_bars, step_bars):
        train_df = df.iloc[start_idx : start_idx + train_bars]
        test_df = df.iloc[start_idx + train_bars : start_idx + train_bars + step_bars]
        
        strategy = XGBStrategy()
        strategy.train(train_df)
        
        preds = strategy.predict(test_df)
        probs = strategy.predict_proba(test_df)
        
        res = test_df.copy()
        res['signal'] = preds
        res['prob'] = probs
        results.append(res)
        
    return pd.concat(results)

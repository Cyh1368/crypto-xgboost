import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import json
import os
from .base_strategy import BaseStrategy
from ..features.registry import registry

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "tree_method": "hist",
    "device": "cpu", # Change to "cuda" if GPU is available
    "random_state": 42,
}

class XGBStrategy(BaseStrategy):
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
        # We need to shift labels because we want to predict NEXT return
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
        self.feature_names = X.columns.tolist()
        
        if eval_set is not None:
            X_val, y_val = self._prepare_data(eval_set)
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            # Split X into train/val for early stopping if no eval_set provided
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # SHAP-based pruning
        self.prune_features(X_val)

    def prune_features(self, X_val):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_val)
        
        # shap_values might be a list for multiclass, but for binary it's usually just an array
        if isinstance(shap_values, list):
            mean_shap = np.abs(shap_values[1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
            
        importance = pd.Series(mean_shap, index=X_val.columns)
        self.pruned_features = importance[importance >= 1e-4].index.tolist()
        
        # Save pruned features
        os.makedirs("btc_backtester/features", exist_ok=True)
        with open(f"btc_backtester/features/pruned_{self.version}.json", "w") as f:
            json.dump(self.pruned_features, f)
            
        print(f"Pruned features: {len(self.feature_names)} -> {len(self.pruned_features)}")

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
        LONG_THRESH = 0.58
        SHORT_THRESH = 0.42
        
        signal = np.where(prob > LONG_THRESH, 1,
                 np.where(prob < SHORT_THRESH, -1, 0))
        return signal

def walk_forward_cv(df: pd.DataFrame, train_window=90, refit_every=7):
    """
    Implements walk-forward cross-validation.
    train_window: days
    refit_every: days
    """
    # Assuming 15-min bars (96 bars per day)
    bars_per_day = 96
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

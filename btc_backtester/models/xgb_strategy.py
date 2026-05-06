import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from btc_backtester.features.technical import prepare_features

class XGBRegressorStrategy:
    def __init__(self, use_grid_search=False):
        self.use_grid_search = use_grid_search
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
        # Best params from paper (Table II and Section IV)
        self.best_params = {
            'n_estimators': 300,
            'learning_rate': 0.2,
            'max_depth': 4,
            'min_child_weight': 3,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 1,
            'reg_lambda': 0.5
        }

    def train(self, df: pd.DataFrame):
        df_feats = prepare_features(df)
        
        # Target is next close
        df_feats['target'] = df_feats['close'].shift(-1)
        df_feats = df_feats.dropna()
        
        self.feature_cols = [c for c in df_feats.columns if c not in ['target', 'open', 'high', 'low', 'timestamp', 'bids', 'asks', 'funding_rate', 'spot_price', 'open_interest']]
        X = df_feats[self.feature_cols]
        y = df_feats['target']
        
        # Scaling
        X_scaled = self.scaler.fit_transform(X)
        
        if self.use_grid_search:
            param_grid = {
                'n_estimators': [300, 400],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
                'reg_alpha': [0.5, 1],
                'reg_lambda': [0.5, 1]
            }
            grid_search = GridSearchCV(XGBRegressor(), param_grid, scoring='neg_mean_squared_error', cv=3)
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            print(f"Best Params: {grid_search.best_params_}")
        else:
            self.model = XGBRegressor(**self.best_params)
            self.model.fit(X_scaled, y)
            
        return df_feats

    def predict(self, df: pd.DataFrame):
        df_feats = prepare_features(df)
        X = df_feats[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        df_feats['predicted_close'] = self.model.predict(X_scaled)
        
        # Signal: Buy if predicted close > current close
        df_feats['signal'] = np.where(df_feats['predicted_close'] > df_feats['close'], 1, -1)
        
        # Probability/Confidence could be mapped from the magnitude of change
        df_feats['prob'] = (df_feats['predicted_close'] - df_feats['close']) / df_feats['close']
        
        return df_feats

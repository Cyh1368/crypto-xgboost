import pandas as pd
from typing import Callable, Dict, List

class FeatureRegistry:
    def __init__(self):
        self._features: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    def register(self, name: str):
        """Decorator to register a feature function."""
        def decorator(func: Callable[[pd.DataFrame], pd.Series]):
            self._features[name] = func
            return func
        return decorator

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all registered features and returns a new DataFrame."""
        feature_df = pd.DataFrame(index=df.index)
        for name, func in self._features.items():
            feature_df[name] = func(df)
        return feature_df

    def get_feature_names(self) -> List[str]:
        return list(self._features.keys())

# Global registry instance
registry = FeatureRegistry()

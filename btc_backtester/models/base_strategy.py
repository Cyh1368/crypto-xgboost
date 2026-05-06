from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame):
        """Trains the model using the provided DataFrame."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the signal (-1, 0, 1) for each row in the DataFrame."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the probability of up move for each row in the DataFrame."""
        pass

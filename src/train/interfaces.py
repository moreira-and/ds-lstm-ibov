from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class ITrainerRunner(ABC):
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        pass

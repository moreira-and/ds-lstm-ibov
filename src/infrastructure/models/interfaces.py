from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class IModelTemplate(ABC):
    
    @abstractmethod
    def train(self, df: pd.Series) -> Any:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Any:
        pass

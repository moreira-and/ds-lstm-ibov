from abc import ABC, abstractmethod
from typing import  Optional, Tuple, Union
import pandas as pd
import numpy as np

class ICleanStrategy(ABC):
    @abstractmethod
    def clear(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Abstract method that should be implemented by any class that inherits from CleanHandler.
        This method should be used for cleaning the data (X) and optionally the target (y).
        """
        pass
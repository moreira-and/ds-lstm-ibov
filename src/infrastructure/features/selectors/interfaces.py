"""
Interface module defining abstract base classes for each stage in a data processing pipeline.

Each interface represents a single responsibility such as data cleaning, selection, transformation, and postprocessing.
These should be implemented by concrete strategy classes adhering to the defined signatures, enabling modular,
extensible, and testable pipelines.

Author: [Your Name]
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np

class ISelectStrategy(ABC):
    
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """
        Defines an interface for selecting relevant features or samples from a dataset.
        """
        pass

    @abstractmethod
    def select(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Defines an interface for selecting relevant features or samples from a dataset.
        """
        pass

    def fit_select(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        self.fit(X, y)
        return self.select(X, y)
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

class IPreprocessorStrategy(ABC):
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """
        Fits the transformation strategy on the dataset.
        """
        pass

    @abstractmethod
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Applies the transformation to the dataset.
        """
        pass

    @abstractmethod
    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Applies the transformation to the dataset.
        """
        pass

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        self.fit(X, y)
        return self.transform(X, y)

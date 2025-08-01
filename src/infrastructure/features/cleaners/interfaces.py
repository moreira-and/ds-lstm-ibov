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




class ICleanStrategy(ABC):
    @abstractmethod
    def clear(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Defines an interface for cleaning raw datasets.

        Parameters:
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix to be cleaned.
        y : pd.Series | np.ndarray | None, default=None
            Optional target variable.

        Returns:
        -------
        Tuple containing cleaned X and y.
        """
        pass
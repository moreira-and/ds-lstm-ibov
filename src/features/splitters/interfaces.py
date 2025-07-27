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

class ISplitterStrategy(ABC):
    @abstractmethod
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple:
        """
        Defines an interface to split data into train/test sets or time-based folds.
        """
        pass

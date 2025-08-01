from config import logger
from ..interfaces import ISelectStrategy

from typing import Optional, Tuple, Union
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np




# Class to remove features with low variance
class LowVarianceSelector(ISelectStrategy):
    def __init__(self, threshold: float = 1e-4):
        """
        Initialize the class with a threshold for variance.

        :param threshold: The variance threshold below which features will be removed.
        """
        self._threshold = threshold

    def select(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        
        logger.info(f"Executing {self.__class__.__name__} with threshold={self._threshold}...")

        # Ensure that X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure that y is a DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        # Apply VarianceThreshold to remove low-variance features
        selector = VarianceThreshold(threshold=self._threshold)
        X_new = selector.fit_transform(X)

        # Get selected columns
        selected_columns = X.columns[selector.get_support()]
        
        X_new = pd.DataFrame(X_new, columns=selected_columns, index=X.index)
        
        # Get dropped columns
        dropped_columns = X.columns.difference(selected_columns)
        logger.info(f"Columns removed due to low variance: {list(dropped_columns)}")

        return X_new, y
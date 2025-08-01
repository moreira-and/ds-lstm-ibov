from config import logger
from ..interfaces import ISelectStrategy

from typing import Optional, Tuple, Union

import pandas as pd
import numpy as np


# Class to remove features that are highly correlated
class HighCorrelationSelector(ISelectStrategy):
    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize the class with a correlation threshold.

        :param correlation_threshold: The threshold above which features will be removed due to high correlation.
        """
        self._correlation_threshold = correlation_threshold

    def select(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        
        logger.info(f"Executing {self.__class__.__name__} with threshold={self._correlation_threshold}...")
        
        # Ensure that X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure that y is a DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        # Compute correlation matrix and select highly correlated features to drop
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > self._correlation_threshold)]
        logger.info(f"Columns removed due to high correlation: {to_drop}")

        X_new = X.drop(columns=to_drop)

        return X_new, y
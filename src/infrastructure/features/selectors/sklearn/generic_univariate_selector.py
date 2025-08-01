from config import logger
from ..interfaces import ISelectStrategy

from typing import Optional, Tuple, Union
from sklearn.feature_selection import GenericUnivariateSelect

import pandas as pd
import numpy as np




# Class to select features based on univariate statistical tests (e.g., ANOVA F-test)
class GenericUnivariateSelector(ISelectStrategy):
    '''
    Score functions (f_classif, f_regression, chi2, etc.)
    mode ("percentile","k_best","fpr","fdr","fwe")
    param (threshold or number of features)
    '''
    def __init__(self, score_func, mode=None, param=None):
        """
        Initialize the class with the score function, mode, and parameter for feature selection.

        :param score_func: The scoring function to evaluate the features (e.g., f_classif, f_regression).
        :param mode: The mode of feature selection (e.g., percentile, k_best, etc.).
        :param param: The parameter for the selected mode (e.g., percentile value or number of features).
        """
        self._score_func = score_func
        self._mode = mode or 'percentile'
        self._param = param or 20

    def select(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        
        # Ensure that X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure that y is a DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        # Apply GenericUnivariateSelect for feature selection
        selector = GenericUnivariateSelect(score_func=self._score_func, mode=self._mode, param=self._param)
        selector.fit(X, y)

        # Get selected columns
        selected_columns = X.columns[selector.get_support()]
        X_new = X[selected_columns]

        # Get dropped columns
        dropped_columns = X.columns.difference(selected_columns)
        logger.info(f"Columns removed by {self._score_func.__name__}: {list(dropped_columns)}")

        return X_new, y

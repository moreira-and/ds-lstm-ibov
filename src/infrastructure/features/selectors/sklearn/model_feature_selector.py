from config import logger
from ..interfaces import ISelectStrategy

from typing import Optional, Tuple, Union
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np


# Class to apply sequential feature selection
class ModelFeatureSelector(ISelectStrategy):
    def __init__(self, model=None, n_features_to_select=None, direction=None):
        """
        Initialize the class with the model and parameters for sequential feature selection.

        :param model: The model used for feature selection (default is LinearRegression).
        :param n_features_to_select: The number of features to select.
        :param direction: The direction for selection ('forward' or 'backward').
        """
        self._model = model or LinearRegression()
        self._n_features_to_select = n_features_to_select or 5
        self._direction = direction or 'forward'

    def select(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        
        logger.info(f"Executing {self.__class__.__name__} with model {self._model.__class__.__name__}...")

        # Ensure that X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure that y is a DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        # Apply SequentialFeatureSelector for feature selection
        selector = SequentialFeatureSelector(self._model, 
                                             n_features_to_select=self._n_features_to_select, 
                                             direction=self._direction)
        selector.fit(X, y)

        # Get selected columns
        selected_columns = X.columns[selector.get_support()]
        X_new = X[selected_columns]

        # Get dropped columns
        dropped_columns = X.columns.difference(selected_columns)
        logger.info(f"Columns removed by 'SequentialFeatureSelector' for model {self._model.__class__.__name__}: {list(dropped_columns)}")

        return X_new, y

from config import logger
from ..interfaces import ICleanStrategy

from typing import Optional, Tuple, Union
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np


'''
#https://scikit-learn.org/stable/api/sklearn.impute.html
import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
imp_mean.transform(X)

import numpy as np
from sklearn.impute import KNNImputer
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(X)


import numpy as np
from sklearn.impute import MissingIndicator
X1 = np.array([[np.nan, 1, 3],
               [4, 0, np.nan],
               [8, 1, 0]])
X2 = np.array([[5, 1, np.nan],
               [np.nan, 2, 3],
               [2, 4, 0]])
indicator = MissingIndicator()
indicator.fit(X1)
X2_tr = indicator.transform(X2)
X2_tr
'''

# Class to apply sequential feature selection
class ModelFeatureSelector(ICleanStrategy):
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

    def clear(self, 
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

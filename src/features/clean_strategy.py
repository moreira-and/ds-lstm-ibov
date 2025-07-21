from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import pandas as pd
import numpy as np

from sklearn.feature_selection import (GenericUnivariateSelect, VarianceThreshold, SequentialFeatureSelector)
from sklearn.linear_model import LinearRegression

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



'''    # Supondo df como seu DataFrame
    target_cols = [col for col in df_raw.columns if asset in col]
    target_col = [col for col in target_cols if asset_focus in col]
    
    df_raw = df_raw.dropna(subset=target_col).sort_index()
    df_raw = df_raw.ffill().bfill()
    df_raw = df_raw.pct_change(periods=1,fill_method=None)
    df_raw = df_raw.replace([np.inf, -np.inf], 0).fillna(0)'''

# Abstract base class for data cleaning handlers
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
        raise NotImplementedError("Implement in subclass")

# Pipeline for cleaning data by applying multiple cleaning steps sequentially
class CleanPipeline(ICleanStrategy):
    '''
    This class allows for applying a series of cleaning steps to the data in sequence.
    '''
    def __init__(self, steps: list):
        """
        Initializes the pipeline with a list of cleaning steps.

        :param steps: A list of CleanHandler objects that will be applied sequentially.
        """
        self.steps = steps

    def clear(self, X, y=None):
        """
        Apply all the cleaning steps sequentially.

        :param X: Data to be cleaned (features).
        :param y: Optional target data (labels).
        :return: Cleaned data (X) and target data (y).
        """
        for step in self.steps:
            X, y = step.clear(X, y)
        return X, y

# Class to handle missing values in the data by filling forward and backward
class CleanMissingValues(ICleanStrategy):
    def clear(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        
        logger.info(f"Executing {self.__class__.__name__}...")

        # Ensure that X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure that y is a DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        X.sort_index(inplace=True)
        y.sort_index(inplace=True)

        assert (X.index == y.index).all(), "X and y indices do not match after sorting"

        # Fill missing values using forward and backward filling
        X_new = X.ffill().bfill()

        y_new = None
        if y is not None:
            # Ensure that y is a Series for safe manipulation
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            y_new = y.ffill().bfill()

        return X_new, y_new

# Class to remove features with low variance
class CleanLowVariance(ICleanStrategy):
    def __init__(self, threshold: float = 1e-4):
        """
        Initialize the class with a threshold for variance.

        :param threshold: The variance threshold below which features will be removed.
        """
        self._threshold = threshold

    def clear(self, 
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

# Class to remove features that are highly correlated
class CleanHighCorrelation(ICleanStrategy):
    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize the class with a correlation threshold.

        :param correlation_threshold: The threshold above which features will be removed due to high correlation.
        """
        self._correlation_threshold = correlation_threshold

    def clear(self, 
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

# Class to select features based on univariate statistical tests (e.g., ANOVA F-test)
class CleanGenericUnivariate(ICleanStrategy):
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

    def clear(self, 
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

# Class to apply sequential feature selection
class CleanSequential(ICleanStrategy):
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

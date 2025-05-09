from abc import ABC, abstractmethod

import numpy as np
from src.config import logger
from src.utils.features.postprocessor_strategy import DefaultPostprocessor

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class PreprocessorStrategy(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass

    @abstractmethod
    def fit_transform(self, X, y=None):
        pass

    @abstractmethod
    def get_postprocessor(self):
        pass

class DefaultPreprocessor(PreprocessorStrategy):
    def __init__(self, 
                 numeric_transformer=StandardScaler(),
                 categorical_transformer=OneHotEncoder(sparse_output=False, handle_unknown='ignore')):
        self._numeric_transformer = numeric_transformer
        self._categorical_transformer = categorical_transformer

        # Define a ColumnTransformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('numeric', self._numeric_transformer, make_column_selector(dtype_include="number")),
                ('categorical', self._categorical_transformer, make_column_selector(dtype_include="object"))
            ],
            remainder='passthrough'
        )

    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.column_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.column_transformer.fit_transform(X, y)

    def get_params(self, deep=True):
        return self.column_transformer.get_params(deep)

    def set_params(self, **params):
        self.column_transformer.set_params(**params)
        return self

    def get_postprocessor(self):
        return DefaultPostprocessor(self.column_transformer)
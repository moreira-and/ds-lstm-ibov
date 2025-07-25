from abc import ABC, abstractmethod

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.utils.features.postprocessor_strategy import (BlankPostprocessor, IPostprocessorStrategy,DefaultRnnPostprocessor)

class DefaultRnnTransformStrategy(ITransformStrategy):
    def __init__(
        self, 
        numeric_transformer=RobustScaler(),
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
        return self.column_transformer.transform(X), y
    
    def get_postprocessor(self, y_train) -> IPostprocessorStrategy:
        return DefaultRnnPostprocessor(self._numeric_transformer,y_train=y_train)
    
    def get_feature_names(self, input_features = None):
        return self.column_transformer.get_feature_names_out(input_features=input_features)

class BlankTransformStrategy(ITransformStrategy):
    def __init__(self, X_column_names=None,y_column_names=None):
        self.X_column_names = X_column_names
        self.y_column_names = y_column_names

    def fit(self, X, y=None):
        self.X_column_names = X.columns if hasattr(X, 'columns') else None
        self.y_column_names = y.columns if hasattr(y, 'columns') else None

    def transform(self, X, y=None):
        return np.array(X), np.array(y) if y is not None else None

    def get_feature_names(self):
        return self.X_column_names

    def get_postprocessor(self, y_train) -> IPostprocessorStrategy:
        return BlankPostprocessor(column_names=self.get_feature_names())
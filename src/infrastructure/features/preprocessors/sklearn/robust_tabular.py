from config import logger
from ..interfaces import IPreprocessorStrategy

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import pandas as pd
import numpy as np

class RobustTabularPreprocessor(IPreprocessorStrategy):
    def __init__(self): 
        self._numeric_transformer = RobustScaler()
        self._categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('numeric', self._numeric_transformer, make_column_selector(dtype_include="number")),
                ('categorical', self._categorical_transformer, make_column_selector(dtype_include="object"))
            ],
            remainder='drop'
        )
        self._columns = None
        self._index = None
        self._feature_names_out = None
        self._transformed_data = None

    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        self._columns = X.columns
        self._index = X.index
        self._feature_names_out = self.column_transformer.get_feature_names_out(X.columns)
        return self

    def transform(self, X, y=None):
        transformed = self.column_transformer.transform(X)
        self._transformed_data = transformed[-1:]
        return pd.DataFrame(transformed, columns=self._feature_names_out, index=X.index), y

    def inverse_transform(self, X_partial, y=None):
        """
        Recupera os valores inversos usando o transformado completo e retorna apenas as colunas relevantes.
        """
        if self._transformed_data is None:
            raise ValueError("Você precisa chamar transform() antes de usar inverse_transform().")

        if isinstance(X_partial, pd.DataFrame):
            partial_cols = X_partial.columns
        else:
            raise ValueError("X_partial precisa ser um DataFrame com nomes de colunas.")

        col_indices = [list(self._feature_names_out).index(col) for col in partial_cols]
        X_full = self._transformed_data.copy()
        X_placeholder = np.zeros_like(X_full)
        X_placeholder[:, col_indices] = X_partial.values

        inversed = self.column_transformer.inverse_transform(X_placeholder)
        return pd.DataFrame(inversed, columns=self._columns, index=self._index)[partial_cols], y

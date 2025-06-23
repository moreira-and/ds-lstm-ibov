from abc import ABC, abstractmethod

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.utils.features.postprocessor_strategy import (BlankPostprocessor, PostprocessorStrategy,DefaultLstmPostprocessor)

class TransformStrategy(ABC):
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
    def get_feature_names(self):
        pass

    @abstractmethod
    def get_postprocessor(self,y_train) -> PostprocessorStrategy:
        pass

class DefaultLstmTransformStrategy(TransformStrategy):
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

    def fit_transform(self, X, y=None):
        return self.column_transformer.fit_transform(X, y)
    
    def get_postprocessor(self, y_train) -> PostprocessorStrategy:
        return DefaultLstmPostprocessor(self._numeric_transformer,y_train=y_train)
    
    def get_feature_names(self, input_features = None):
        return self.column_transformer.get_feature_names_out(input_features=input_features)

class BlankTransformStrategy(TransformStrategy):
    def __init__(self, X_column_names=None,y_column_names=None):
        self.X_column_names = X_column_names
        self.y_column_names = y_column_names

    def fit(self, X, y=None):
        self.X_column_names = X.columns if hasattr(X, 'columns') else None
        self.y_column_names = y.columns if hasattr(y, 'columns') else None

    def transform(self, X, y=None):
        return np.array(X), np.array(y) if y is not None else None

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names(self):
        return self.X_column_names

    def get_postprocessor(self, y_train) -> PostprocessorStrategy:
        return BlankPostprocessor(column_names=self.get_feature_names())
    

'''
import numpy as np

def rolling_inverse_transform(X_norm, min_vals, max_vals):
    """
    Reverte a normalização feita com rolling_normalize_with_params.

    Parâmetros:
    -----------
    X_norm : np.ndarray
        Dados normalizados (n_amostras, janela, n_features)
    min_vals : np.ndarray
        Mínimos salvos na normalização (n_amostras, n_features)
    max_vals : np.ndarray
        Máximos salvos na normalização (n_amostras, n_features)

    Retorna:
    --------
    X_original : np.ndarray
        Dados na escala original
    """
    X_orig = np.empty_like(X_norm)
    denom = max_vals - min_vals
    denom[denom == 0] = 1

    for i in range(X_norm.shape[0]):
        X_orig[i] = X_norm[i] * denom[i] + min_vals[i]

    return X_orig

def rolling_normalize_with_params(X):
    """
    Normaliza cada janela individualmente (MinMax por janela) e armazena min/max.

    Parâmetros:
    -----------
    X : np.ndarray
        Array de shape (n_amostras, janela, n_features)

    Retorna:
    --------
    X_norm : np.ndarray
        Dados normalizados
    min_vals : np.ndarray
        Mínimos por janela e por feature — shape (n_amostras, n_features)
    max_vals : np.ndarray
        Máximos por janela e por feature — shape (n_amostras, n_features)
    """
    X_norm = np.empty_like(X)
    min_vals = X.min(axis=1)  # shape (n, features)
    max_vals = X.max(axis=1)

    denom = max_vals - min_vals
    denom[denom == 0] = 1  # evita divisão por zero

    for i in range(X.shape[0]):
        X_norm[i] = (X[i] - min_vals[i]) / denom[i]

    return X_norm, min_vals, max_vals


# X: shape (n_amostras, janela, n_features)
X_norm, X_min, X_max = rolling_normalize_with_params(X)

# Após treino e predict:
X_inv = rolling_inverse_transform(X_norm, X_min, X_max)

'''
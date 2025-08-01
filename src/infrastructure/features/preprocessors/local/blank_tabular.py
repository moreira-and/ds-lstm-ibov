from config import logger
from ..interfaces import IPreprocessorStrategy

import pandas as pd
import numpy as np

class BlankTabularPreprocessor(IPreprocessorStrategy):
    def __init__(self):
        self.X_column_names = None
        self.y_column_names = None

    def fit(self, X, y=None):
        self.X_column_names = X.columns if hasattr(X, 'columns') else None
        self.y_column_names = y.columns if hasattr(y, 'columns') else None

    def transform(self, X, y=None):
        return np.array(X), np.array(y) if y is not None else None
    
    def inverse_transform(self, X, y=None):
        __X = pd.DataFrame(X, columns=self.X_column_names) \
            if len(self.X_column_names) > 0 else pd.DataFrame(X)

        __y = pd.DataFrame(y, columns=self.y_column_names) \
            if y is not None and len(self.y_column_names) > 0 else pd.DataFrame(y) if y is not None else None

        return (__X, __y) if y is not None else (__X, None)        

    def get_feature_names(self):
        return self.X_column_names

    


from config import logger
from ..interfaces import IPostprocessorStrategy,ITransformStrategy

import pandas as pd
import numpy as np

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
    
class BlankPostprocessor(IPostprocessorStrategy):  
    
    def __init__(self,column_names=None):
        self.column_names = column_names

    def inverse_transform(self, y_predicted):
        return pd.DataFrame(y_predicted, columns=self.column_names.values) \
            if len(self.column_names.values) > 0 else pd.DataFrame(y_predicted)
    


from abc import ABC, abstractmethod
from src.config import logger

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


class DefaultPreprocessor(PreprocessorStrategy):
    def __init__(self):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('scaler', StandardScaler(), make_column_selector(dtype_include="number")),
                    ('encoder', OneHotEncoder(), make_column_selector(dtype_include="object"))
                ],
                remainder='passthrough'
            ))
        ])

    def fit(self, X, y=None):
        logger.info("Fitting the transformer on the training dataset...")
        return self.pipeline.fit(X)
    
    def transform(self, X, y=None):
        logger.info("Transforming the dataset using the transformer...")
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        logger.info("Fitting the transformer and transforming the dataset using it...")
        return self.pipeline.fit_transform(X)

    
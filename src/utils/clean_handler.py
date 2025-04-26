from typing import Any
from src.config import PROCESSED_DATA_DIR

from abc import ABC, abstractmethod

from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector

class CleanHandler(ABC):
    @abstractmethod
    def clear(X:Any=None) -> Any:
        pass

class ClearPipeline(CleanHandler):
    def clear(X):
        pass

class ClearMissing(CleanHandler):
    def clear(X):
        pass

class ClearMissing(CleanHandler):
    def clear(X):
        pass

class ClearMissing(CleanHandler):
    def clear(X):
        pass
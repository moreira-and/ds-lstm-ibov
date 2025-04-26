from typing import Any
from src.config import PROCESSED_DATA_DIR

from abc import ABC, abstractmethod

class ModelHandler(ABC):
    @abstractmethod
    def train_test_slipt(X:Any, Y:Any = None) -> Any:
        pass

class LstmHandler(ModelHandler):
    def train_test_slipt(X, Y = None):
        pass
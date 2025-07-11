from abc import ABC, abstractmethod
from src.config import logger

class ISplitterStrategy(ABC):
    @abstractmethod
    def split(self,X, y=None):
        raise NotImplementedError("Implement in subclass")


class SequentialSplitter(ISplitterStrategy):
    def __init__(self, train_size_ratio=0.9):
        self.train_size_ratio = train_size_ratio

    def split(self, X, y=None):
        logger.info("Splitting dataset into training and testing sets...")

        X_train_size = int(len(X) * self.train_size_ratio)

        if y is None:
            return X[:X_train_size], X[X_train_size:]
        
        y_train_size = int(len(y) * self.train_size_ratio)

        return X[:X_train_size], X[X_train_size:], y[:y_train_size], y[X_train_size:]
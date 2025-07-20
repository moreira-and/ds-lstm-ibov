from abc import ABC, abstractmethod
from src.config.config import logger

class ISplitterStrategy(ABC):
    @abstractmethod
    def split(self,X, y=None):
        raise NotImplementedError("Implement in subclass")


class SequentialSplitter(ISplitterStrategy):
    def __init__(self, train_size_ratio=0.95):
        self.train_size_ratio = train_size_ratio

    def split(self, X, y=None):
        
        logger.info("Splitting dataset into training and testing sets...")

        if len(X) == 0:
            raise ValueError("Input data X is empty.")
        
        if not (0 < self.train_size_ratio < 1):
            raise ValueError("train_size_ratio must be between 0 and 1.")

        train_size = int(len(X) * self.train_size_ratio)

        if y is None:
            return X[:train_size], X[train_size:]

        return X[:train_size], X[train_size:], y[:train_size], y[train_size:]
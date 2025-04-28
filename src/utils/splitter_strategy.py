from abc import ABC, abstractmethod


class SplitterStrategy(ABC):
    @abstractmethod
    def split(self,dataset):
        pass


class SequentialSplitter(SplitterStrategy):
    def __init__(self, train_size_ratio=0.9):
        self.train_size_ratio = train_size_ratio

    def split(self, X, y=None):

        X_train_size = int(len(X) * self.train_size_ratio)

        if y is None:
            return X[:X_train_size], X[X_train_size:]
        
        y_train_size = int(len(y) * self.train_size_ratio)

        return X[:X_train_size], X[X_train_size:], y[:y_train_size], y[X_train_size:]
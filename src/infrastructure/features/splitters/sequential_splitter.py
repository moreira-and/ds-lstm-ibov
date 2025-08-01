from config import logger
from .interfaces import ISplitterStrategy

class SequentialLengthSplitter(ISplitterStrategy):
    def __init__(self, test_length=16):
        self.test_length = test_length

    def split(self, X, y=None):
        
        if len(X) == 0:
            raise ValueError("Input data X is empty.")
        
        X.sort_index(inplace=True)

        if y is not None:
            y.sort_index(inplace=True)
            assert (X.index == y.index).all(), "X and y indices do not match after sorting"
        
        logger.info(f"Splitting dataset into training ({X.shape[0]}) and testing ({self.test_length}) sets...")

        if y is None:
            return X[:self.test_length], X[self.test_length:], None, None

        return X[:self.test_length], X[self.test_length:], y[:self.test_length], y[self.test_length:]
from config import logger
from ..interfaces import ISplitterStrategy

class SequentialSplitter(ISplitterStrategy):
    def __init__(self, test_length=16):
        self.test_length = test_length

    def split(self, X, y=None):
        
        if len(X) == 0:
            raise ValueError("Input data X is empty.")
        
        logger.info(f"Splitting dataset into training ({X.shape[0]}) and testing ({self.test_length}) sets...")

        if y is None:
            return X[:self.test_length], X[self.test_length:]

        return X[:self.test_length], X[self.test_length:], y[:self.test_length], y[self.test_length:]
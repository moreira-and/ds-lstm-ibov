from .interfaces import ICleanStrategy
from typing import List

# Pipeline for cleaning data by applying multiple cleaning steps sequentially
class CleanPipeline(ICleanStrategy):
    '''
    This class allows for applying a series of cleaning steps to the data in sequence.
    '''
    def __init__(self, steps: List[ICleanStrategy]):
        """
        Initializes the pipeline with a list of cleaning steps.

        :param steps: A list of CleanHandler objects that will be applied sequentially.
        """
        self.steps = steps

    def clear(self, X, y=None):
        """
        Apply all the cleaning steps sequentially.

        :param X: Data to be cleaned (features).
        :param y: Optional target data (labels).
        :return: Cleaned data (X) and target data (y).
        """
        for step in self.steps:
            X, y = step.clear(X, y)
        return X, y
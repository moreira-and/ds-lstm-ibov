
from abc import ABC, abstractmethod


class TransformerBuilder(ABC):

    @abstractmethod
    def transform(self,X):
        pass  


class PreprocessorAndTimeSeries(TransformerBuilder):
    def __init__(self,preprocessor,generator):
        self.preprocessor = preprocessor
        self.generator = generator

    def transform(self, X):
        X_transformed = self.preprocessor.transform(X = X)
        return self.generator.generate(X_transformed)


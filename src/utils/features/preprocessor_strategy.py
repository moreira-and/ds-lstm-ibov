from abc import ABC, abstractmethod

class PreprocessorStrategy(ABC):
    @abstractmethod
    def transform(self, X, y=None):
        pass

class DefaultLstmPreprocessor(PreprocessorStrategy):

    def __init__(self, transformer, generator):
        self._transformer = transformer
        self._generator = generator

    def transform(self, X, y=None):
        self._transformer.fit(X, y)
        return self._generator.transform(X)
from abc import ABC, abstractmethod

class DefaultRnnPreprocessor(IPreprocessorStrategy):

    def __init__(self, transformer, generator):
        self._transformer = transformer
        self._generator = generator

    def transform(self, X, y=None):
        X_transformed,y_transformed = self._transformer.transform(X, y)
        return self._generator.generate(X_transformed)
from abc import ABC, abstractmethod

class PredictorBuilder(ABC):
    @abstractmethod
    def predict(self, X):
        pass
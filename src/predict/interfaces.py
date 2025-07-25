from abc import ABC, abstractmethod

class IPredictTemplate(ABC):
    @abstractmethod
    def predict(self, X):
        pass
from abc import ABC, abstractmethod

class ITrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass
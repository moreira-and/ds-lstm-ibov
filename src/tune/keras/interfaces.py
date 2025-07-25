from abc import ABC, abstractmethod
from keras_tuner import HyperParameters

class ITunerBuilder(ABC):
    @abstractmethod
    def get_model(self,hp: HyperParameters):
        pass

    @abstractmethod
    def build_tuner(self):
        pass

class ISearchStrategy(ABC):
    @abstractmethod
    def search(self, model, X_train, y_train):
        pass


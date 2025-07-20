from abc import ABC, abstractmethod

class ITrainerRunner(ABC):
    @abstractmethod
    def run(self, X_train, y_train):
        pass

class IModelBuilder(ABC):
    @abstractmethod
    def build_model(self):
        pass

class ICallbacksStrategy(ABC):
    @staticmethod
    @abstractmethod
    def get():
        pass

class ICompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class IMetricStrategy(ABC):
    @abstractmethod
    def get_metrics(self):
        pass

class ITrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass
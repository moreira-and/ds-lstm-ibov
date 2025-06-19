from src.config import logger

from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC

class CompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class ClassificationCompileStrategy(CompileStrategy):
    def __init__(self, optimizer = Adam(learning_rate=0.01), loss = 'binary_crossentropy', metrics = ['accuracy', Precision(), Recall(), AUC()]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compile(self, model):
        try:
            model.compile(
                optimizer = self.optimizer,
                loss = self.loss,
                metrics = self.metrics
            )
        except Exception as e:
            logger.error(f'Error compilling {self.__class__.__name__}: {e}')

class RegressionCompileStrategy(CompileStrategy):
    def __init__(self, optimizer = Adam(learning_rate=0.01), loss = 'mse', metrics = ['mae','mape']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compile(self, model):
        try:
            model.compile(
                optimizer = self.optimizer,
                loss = self.loss,
                metrics = self.metrics
            )
        except Exception as e:
            logger.error(f'Error compilling {self.__class__.__name__}: {e}')

from src.config import logger
from src.utils.train.metric_strategy import ClassificationMetricStrategy, RegressionMetricStrategy

from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber



class CompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class ClassificationCompileStrategy(CompileStrategy):
    def __init__(self, optimizer = Adam(learning_rate=0.01), loss = 'binary_crossentropy', metrics = ClassificationMetricStrategy().get_metrics()):
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
    def __init__(self, optimizer = Adam(learning_rate=0.001), loss = Huber(delta=1.0), metrics = RegressionMetricStrategy().get_metrics()):
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
            raise

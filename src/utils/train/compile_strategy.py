from src.config import logger
from src.utils.train.metric_strategy import ClassificationMetricStrategy, RegressionMetricStrategy

from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber



class CompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        raise NotImplementedError("Implement in subclass")

class ClassificationCompileStrategy(CompileStrategy):
    def __init__(self, loss = 'binary_crossentropy', optimizer_fn = None, metrics = None):
        self.loss = loss
        self.optimizer = optimizer_fn or Adam(learning_rate=0.01)
        self.metrics = metrics or ClassificationMetricStrategy().get_metrics()

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
    def __init__(self, optimizer_fn=None, loss=None, metrics=None):
        self.optimizer_fn = optimizer_fn or (lambda: Adam(learning_rate=0.001))
        self.loss = loss or Huber(delta=1.0)
        self.metrics = metrics or RegressionMetricStrategy().get_metrics()

    def compile(self, model):
        try:
            optimizer = self.optimizer_fn()
            model.compile(
                optimizer=optimizer,
                loss=self.loss,
                metrics=self.metrics
            )
        except Exception as e:
            logger.error(f'Error compilling {self.__class__.__name__}: {e}')
            raise
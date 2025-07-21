from src.config import logger
from src.train.keras.interfaces import ICompileStrategy
from train.keras.metric_strategy import RegressionMetrics

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class RegressionCompile(ICompileStrategy):
    def __init__(self, optimizer_fn=None, loss=None, metrics=None):
        self.optimizer_fn = optimizer_fn or (lambda: Adam(learning_rate=0.001))
        self.loss = loss or Huber(delta=2.0)
        self.metrics = metrics or RegressionMetrics().get_metrics()

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
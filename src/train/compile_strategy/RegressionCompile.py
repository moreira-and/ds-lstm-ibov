from src.config import logger
from src.utils.train.metric_strategy import RegressionMetricStrategy

from src.train.interfaces import ICompileStrategy

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class RegressionCompile(ICompileStrategy):
    def __init__(self, optimizer_fn=None, loss=None, metrics=None):
        self.optimizer_fn = optimizer_fn or (lambda: Adam(learning_rate=0.001))
        self.loss = loss or Huber(delta=2.0)
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
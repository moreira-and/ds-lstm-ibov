from config import logger
from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper
from .interfaces import ICompileStrategy
from .measurers import regression_metrics

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class RegressionCompile(ICompileStrategy):
    def __init__(self, optimizer_fn=None, loss=None, metrics=None):

        params = ConfigWrapper(TRAIN_PARAMS_FILE)
        self.optimizer = optimizer_fn or Adam(
            learning_rate= params.get("learning_rate"),
            clipnorm= params.get("clipnorm")
            )
        
        self.loss = loss or Huber(delta=2.0)
        self.metrics = metrics or regression_metrics()

    def compile(self, model):
        try:
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metrics
            )
        except Exception as e:
            logger.exception(f'Error compilling {self.__class__.__name__}: {e}')
            raise
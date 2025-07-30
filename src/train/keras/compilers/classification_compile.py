from config import logger
from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper
from .interfaces import ICompileStrategy
from .measurers import classification_metrics

from tensorflow.keras.optimizers import Adam


class ClassificationCompile(ICompileStrategy):
    def __init__(self, loss = 'binary_crossentropy', optimizer_fn = None, metrics = None):
        self.loss = loss        
        self.metrics = metrics or classification_metrics()

        params = ConfigWrapper(TRAIN_PARAMS_FILE)
        self.optimizer = optimizer_fn or Adam(
            learning_rate=params.get("learning_rate"),
            clipnorm=params.get("clipnorm")
            )

    def compile(self, model):
        try:
            model.compile(
                optimizer = self.optimizer,
                loss = self.loss,
                metrics = self.metrics
            )
        except Exception as e:
            logger.exception(f'Error compilling {self.__class__.__name__}: {e}')
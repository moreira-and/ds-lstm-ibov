from src.config import logger
from train.interface import ICompileStrategy

from train.metrics_strategy import ClassificationMetricStrategy


from tensorflow.keras.optimizers import Adam


class ClassificationCompile(ICompileStrategy):
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
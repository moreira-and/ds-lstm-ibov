from abc import ABC, abstractmethod
from src.config import logger
from src.utils.train.callbacks_strategy import RegressionCallbacksStrategy

class TrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass

class RegressionTrainStrategy(TrainStrategy):
    def __init__(self, epochs= 100, batch_size= 16, validation_split= 0.3, callbacks = RegressionCallbacksStrategy.get()):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.callbacks = callbacks

    def train(self, model, X_train, y_train):
        try:
            return model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks = self.callbacks
            )
        except Exception as e:
            logger.error(f'Error training {self.__class__.__name__}: {e}')
            return model



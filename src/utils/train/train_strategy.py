from abc import ABC, abstractmethod
from src.config import logger
from src.utils.train.callbacks_strategy import DefaultCallbacksStrategy

class TrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass

class BasicTrainStrategy(TrainStrategy):
    def __init__(self, epochs= None, batch_size= None, validation_split= None, callbacks = None):
        self.epochs = epochs or 100
        self.batch_size = batch_size or 16
        self.validation_split = validation_split or 0.3
        self.callbacks = callbacks or DefaultCallbacksStrategy.get()

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



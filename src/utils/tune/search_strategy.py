from abc import ABC, abstractmethod

from src.config import logger

from keras import callbacks

from src.utils.train.callbacks_strategy import CallbacksStrategy, RegressionCallbacksStrategy

class SearchStrategy(ABC):
    @abstractmethod
    def search(self, model, X_train, y_train):
        pass

class RegressionTuneStrategy(SearchStrategy):
    def __init__(self, epochs : int = 200, batch_size : int = 16, validation_len : int = 20, callbacks : callbacks = RegressionCallbacksStrategy.get()):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_len = validation_len
        self.callbacks = callbacks

    def search(self, tuner, X, y):
        try:
            # Separação manual por tempo
            X_train, X_val = X[:-self.validation_len], X[-self.validation_len:]
            y_train, y_val = y[:-self.validation_len], y[-self.validation_len:]

            return tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=False,
                callbacks = self.callbacks
            )
        
        except Exception as e:
            logger.error(f'Error training {self.__class__.__name__}: {e}')
            raise e
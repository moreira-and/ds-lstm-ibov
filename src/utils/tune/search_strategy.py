from abc import ABC, abstractmethod

from src.config import logger

from keras import callbacks
import numpy as np
from src.utils.train.callbacks_strategy import CallbacksStrategy, RegressionCallbacksStrategy

class SearchStrategy(ABC):
    @abstractmethod
    def search(self, model, X_train, y_train):
        raise NotImplementedError("Implement in subclass")

class RegressionTuneStrategy(SearchStrategy):
    def __init__(self, epochs : int = 200, batch_size : int = 16, validation_len : int = 32, callbacks : callbacks = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_len = validation_len
        self.callbacks = callbacks or RegressionCallbacksStrategy.get()

        if self.validation_len < self.batch_size:
            raise ValueError(f"Validation length ({self.validation_len}) must be >= batch size ({self.batch_size})")


    def search(self, tuner, X: np.ndarray, y:np.ndarray):
        try:
            
            if X is None or y is None:
                raise ValueError("X or y is None. Ensure the dataset is properly loaded.")

            # Separação manual por tempo
            X_train, X_val = X[:-self.validation_len], X[-self.validation_len:]
            y_train, y_val = y[:-self.validation_len], y[-self.validation_len:]

            logger.debug(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

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
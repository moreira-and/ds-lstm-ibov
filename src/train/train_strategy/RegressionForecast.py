from train.interface import ITrainStrategy
from src.config import logger

from train.callbacks_strategy import RegressionCallbacks

class RegressionForecast(ITrainStrategy):
    def __init__(self, epochs= 200, batch_size= 16, validation_len = 32, callbacks = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_len = validation_len
        self.callbacks = callbacks or RegressionCallbacks.get()

        if self.validation_len < self.batch_size:
            raise ValueError(f"Validation length ({self.validation_len}) must be >= batch size ({self.batch_size})")

    def train(self, model, X, y):
        try:

            if X is None or y is None:
                raise ValueError("X or y is None. Ensure the dataset is properly loaded.")
            
            # Separação manual por tempo
            X_train, X_val = X[:-self.validation_len], X[-self.validation_len:]
            y_train, y_val = y[:-self.validation_len], y[-self.validation_len:]

            return model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=False,
                callbacks = self.callbacks
            )
        
        except Exception as e:
            logger.error(f'Error training {self.__class__.__name__}: {e}')
            raise
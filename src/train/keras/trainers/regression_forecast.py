from config import logger
from config.paths import TRAIN_PARAMS_FILE

from utils import ConfigWrapper

from .interfaces import ITrainStrategy
from .callbacks import RegressionCallbacks

class RegressionForecast(ITrainStrategy):
    def __init__(self, callbacks = None):
        self.callbacks = callbacks or RegressionCallbacks()

        params = ConfigWrapper(TRAIN_PARAMS_FILE)
        self.epochs = params.get("epochs")
        self.batch_size = params.get("batch_size")
        self.validation_len = params.get("validation_len")        

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
                shuffle=False,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks = self.callbacks
            )
        
        except Exception as e:
            logger.error(f'Error training {self.__class__.__name__}: {e}')
            raise
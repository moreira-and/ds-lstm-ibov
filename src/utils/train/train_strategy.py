from abc import ABC, abstractmethod
from src.config import logger
from src.utils.train.callbacks_strategy import RegressionCallbacksStrategy

class TrainStrategy(ABC):
    @abstractmethod
    def train(self, model, X_train, y_train):
        pass

class RegressionTrainStrategy(TrainStrategy):
    def __init__(self, epochs= 200, batch_size= 16, validation_len = 20, callbacks = RegressionCallbacksStrategy.get()):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_len = validation_len
        self.callbacks = callbacks

    def train(self, model, X, y):
        try:
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
            return model


class RegressionTuneStrategy(TrainStrategy):
    def __init__(self, epochs= 200, batch_size= 16, validation_len = 20, callbacks = RegressionCallbacksStrategy.get()):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_len = validation_len
        self.callbacks = callbacks

    def train(self, tuner, X, y):
        try:
            # Separação manual por tempo
            X_train, X_val = X[:-self.validation_len], X[-self.validation_len:]
            y_train, y_val = y[:-self.validation_len], y[-self.validation_len:]

            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=False,
                callbacks = self.callbacks
            )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            return tuner.hypermodel.build(best_hps)
        
        except Exception as e:
            logger.error(f'Error training {self.__class__.__name__}: {e}')
            raise e
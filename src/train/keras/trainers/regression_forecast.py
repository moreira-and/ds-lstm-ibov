from config import logger
from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper

from .interfaces import ITrainStrategy
from .callbacks import regression_callbacks
from .generators import sliding_window_generator

from tf.keras.callbacks import History

class RegressionForecast(ITrainStrategy):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or regression_callbacks()

        params = ConfigWrapper(TRAIN_PARAMS_FILE)
        self.epochs = params.get("epochs")
        self.batch_size = params.get("batch_size")
        self.validation_len = params.get("validation_len")
        self.sequence_length = params.get("sequence_length")
        self.target_cols = params.get("targets")  # lista de strings

        if not self.target_cols or not isinstance(self.target_cols, list):
            raise ValueError("Config 'targets' must be a non-empty list of column names.")

        if self.validation_len < self.batch_size:
            raise ValueError(
                f"Validation length ({self.validation_len}) must be >= batch size ({self.batch_size})"
            )
        
        if self.validation_len < self.sequence_length:
            raise ValueError(
                f"Validation length ({self.validation_len}) must be >= sequence_length ({self.sequence_length})"
            )

    def train(self, model, df) -> History:
        try:
            if df is None or len(df) == 0:
                raise ValueError("Input DataFrame is None or empty.")

            split_idx = len(df) - self.validation_len

            df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]
            y_train = df_train[self.target_cols]
            y_val = df_val[self.target_cols]

            train_gen = sliding_window_generator(df_train)
            val_gen = sliding_window_generator(df_val)

            return model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=self.epochs,
                callbacks=self.callbacks,
                shuffle=False,
                verbose=1
            )

        except Exception as e:
            logger.error(f"Error training {self.__class__.__name__}: {e}")
            raise

from config import logger
from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper

from .interfaces import ITrainStrategy
from .callbacks import regression_callbacks
from .data_loading import sliding_window_generator

import pandas as pd

class RegressionForecast(ITrainStrategy):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or regression_callbacks()

        params = ConfigWrapper(TRAIN_PARAMS_FILE)
        self.epochs = params.get("epochs")
        self.batch_size = params.get("batch_size")
        self.validation_len = params.get("validation_len")
        self.sequence_length = params.get("sequence_length")
        self.target_cols = params.get("targets")  # lista de strings
        self.columns = None

        if not self.target_cols or not isinstance(self.target_cols, list):
            logger.error("Config 'targets' must be a non-empty list of column names.")
            raise

        if self.validation_len < self.batch_size:
            logger.error(
                f"Validation length ({self.validation_len}) must be >= batch size ({self.batch_size})"
            )
            raise
        
        if self.validation_len < self.sequence_length:
            logger.error(
                f"Validation length ({self.validation_len}) must be >= sequence_length ({self.sequence_length})"
            )
            raise

    def train(self, model, df: pd.DataFrame):
        try:
            if df is None or len(df) == 0:
                logger.exception("Input DataFrame is None or empty.")
                raise

            self.columns = df.columns

            split_idx = len(df) - self.validation_len

            df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]

            train_ts = sliding_window_generator(df_train)
            val_ts = sliding_window_generator(df_val)

            return model.fit(
                train_ts,
                validation_data=val_ts,
                epochs=self.epochs,
                callbacks=self.callbacks,
                shuffle=False,
                verbose=1
            )
    
        except Exception as e:
            logger.exception(f"Error training {self.__class__.__name__}: {e}")
            raise

    def predict(self, model, df) -> pd.DataFrame:
        try:
            if df is None or len(df) == 0:
                logger.error("Input DataFrame is None or empty.")
                raise

            if len(df) < self.sequence_length:
                logger.error(
                    f"Input DataFrame length ({len(df)}) must be >= sequence_length ({self.sequence_length})"
                )
                raise

            # Garante que as colunas estão na mesma ordem do treino
            df = df[self.columns]

            pred_ts = sliding_window_generator(df)
            predictions = model.predict(pred_ts, verbose=0)

            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)

            index = df.index[self.sequence_length : self.sequence_length + len(predictions)]

            return pd.DataFrame(predictions, columns=self.target_cols, index=index)

        except Exception as e:
            logger.exception(f"Error predicting with {self.__class__.__name__}: {e}")
            raise



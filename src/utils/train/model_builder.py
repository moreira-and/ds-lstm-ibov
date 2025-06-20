from src.config import logger
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, GRU, Dropout, Dense, LayerNormalization,BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l1, l2

class ModelBuilder(ABC):

    @abstractmethod
    def build_model(self):
        return
    
class RegressionRobustModelBuilder(ModelBuilder):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(self):

        try:
            return Sequential([
                Input(shape=self.input_shape),

                LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
                LayerNormalization(),
                Dropout(0.3),

                GRU(32, return_sequences=False, kernel_regularizer=l2(1e-4)),
                Dropout(0.2),

                Dense(32, activation='relu', kernel_regularizer=l2(1e-5)),
                Dropout(0.2),
                BatchNormalization(),

                Dense(self.output_shape[0], activation='linear')
            ])
        except Exception as e:
            logger.error(f'Error building {self.__class__.__name__}: {e}')
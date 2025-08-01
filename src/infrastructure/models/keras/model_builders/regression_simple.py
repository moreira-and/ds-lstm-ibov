from config import logger
from .interfaces import IModelBuilder

from tensorflow.keras.layers import Input, LayerNormalization, Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

class RegressionSimple(IModelBuilder):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(self):

        try:
            return Sequential([
                Input(shape=self.input_shape),
                
                LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-5),recurrent_dropout=0.3),
                LayerNormalization(),

                Dense(16, activation='relu', kernel_regularizer=l2(1e-5)),
                Dropout(0.2),
                
                Dense(self.output_shape[0], activation='linear')
            ])
        except Exception as e:
            logger.exception(f'Error building {self.__class__.__name__}: {e}')
            raise
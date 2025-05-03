from src.config import logger
from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from keras.regularizers import l2

class ModelBuilder(ABC):

    @abstractmethod
    def build_model(self):
        return
    
class LstmModelBuilder(ModelBuilder):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(self):

        try:
            return Sequential([
                Input(shape=self.input_shape),
                Bidirectional(LSTM(70, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.001))),
                Dropout(0.3),
                LSTM(50, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
                Dropout(0.3),
                LSTM(30, return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
                Dense(self.output_shape[0])
            ])
        except Exception as e:
            logger.error(f'Error building {self.__class__.__name__}: {e}')
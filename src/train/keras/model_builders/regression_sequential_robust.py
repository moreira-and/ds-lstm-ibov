from config import logger
from .interfaces import IModelBuilder

from tensorflow.keras.layers import Input, Conv1D, Dropout, \
    LSTM, TimeDistributed, GRU, Dense, LeakyReLU, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

class RegressionSequentialRobust(IModelBuilder):

    def __init__(self, input_shape, output_shape,kernel_rate = 5,l2_rate = 1e-4,dropout_rate=0.2, conv1D_units = 128, gru_units = 64, mid_dense_units = 32, negative_slope_rate = 0.1,lstm_units = 32, low_dense_units = 16):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_rate = kernel_rate
        self.l2_rate = l2_rate
        self.dropout_rate = dropout_rate
        self.conv1D_units = conv1D_units
        self.gru_units = gru_units
        self.mid_dense_units = mid_dense_units
        self.negative_slope_rate = negative_slope_rate
        self.lstm_units = lstm_units
        self.low_dense_units = low_dense_units

    def build_model(self):

        try:
            return Sequential([
                Input(shape=self.input_shape),

                Conv1D(self.conv1D_units, kernel_size=self.kernel_rate, activation='relu', padding='causal', kernel_regularizer=l2(self.l2_rate)),
                Dropout(self.dropout_rate),
                LayerNormalization(),

                GRU(self.gru_units, return_sequences=True, kernel_regularizer=l2(self.l2_rate),recurrent_dropout=self.dropout_rate),
                Dropout(self.dropout_rate),
                LayerNormalization(),

                TimeDistributed(Dense(self.mid_dense_units, activation=None, kernel_regularizer=l2(self.l2_rate))),
                TimeDistributed(LeakyReLU(negative_slope=self.negative_slope_rate)),
                Dropout(self.dropout_rate),

                LSTM(self.lstm_units, return_sequences=False, kernel_regularizer=l2(self.l2_rate),recurrent_dropout=self.dropout_rate),
                Dropout(self.dropout_rate),

                Dense(self.low_dense_units, activation='relu', kernel_regularizer=l2(self.l2_rate)),
                Dropout(self.dropout_rate),

                Dense(self.output_shape[0], activation='linear')
            ])
        except Exception as e:
            logger.exception(f'Error building {self.__class__.__name__}: {e}')
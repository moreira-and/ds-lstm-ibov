from config import logger
from .interfaces import IModelBuilder

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GRU, LSTM, Dense, Dropout, Concatenate,
    LeakyReLU, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class RegressionMultiInput(IModelBuilder):
    def __init__(
        self,
        input_shape_5min,
        input_shape_weekly,
        gru_units=64,
        lstm_units=32,
        dense_units=32,
        dropout_rate=0.2,
        l2_rate=1e-4,
        num_heads=4,
    ):
        self.input_shape_5min = input_shape_5min
        self.input_shape_weekly = input_shape_weekly
        self.gru_units = gru_units
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_rate = l2_rate
        self.num_heads = num_heads

    def build_model(self):
        try:
            input_5min = Input(shape=self.input_shape_5min, name="input_5min")
            x1 = GRU(self.gru_units, return_sequences=True, kernel_regularizer=l2(self.l2_rate))(input_5min)
            x1 = Dropout(self.dropout_rate)(x1)
            x1 = LayerNormalization()(x1)

            attn_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.gru_units)(x1, x1)
            x1 = tf.keras.layers.Add()([x1, attn_output])
            x1 = LayerNormalization()(x1)
            x1 = GlobalAveragePooling1D()(x1)

            input_weekly = Input(shape=self.input_shape_weekly, name="input_weekly")
            x2 = LSTM(self.lstm_units, return_sequences=False, kernel_regularizer=l2(self.l2_rate))(input_weekly)
            x2 = Dropout(self.dropout_rate)(x2)

            combined = Concatenate()([x1, x2])

            x = Dense(self.dense_units, activation=None, kernel_regularizer=l2(self.l2_rate))(combined)
            x = LeakyReLU()(x)
            x = Dropout(self.dropout_rate)(x)
            output = Dense(1, activation='linear', name="output")(x)

            return Model(inputs=[input_5min, input_weekly], outputs=output)
        
        except Exception as e:
            logger.exception(f'Error building {self.__class__.__name__}: {e}')

# Exemplo de uso:
# builder = MultiResolutionModelBuilder((24, 10), (4, 5))
# model = builder.build_model()
# model.summary()

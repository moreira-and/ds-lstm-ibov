from config import logger
from .interfaces import IModelBuilder

from tensorflow.keras.layers import Input, Conv1D, LayerNormalization, Dropout, \
    Bidirectional, LSTM, GRU, Dense, Add, Multiply, Lambda, Concatenate, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


class RegressionMultiLayers(IModelBuilder):
    def __init__(self, input_shape, output_shape,exo_dim=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.exo_dim = exo_dim  # Exogenous input dimension, if applicable

    @staticmethod
    def attention_layer(inputs):
        # inputs shape: (batch, timesteps, features)
        score = Dense(1, activation='tanh')(inputs)  # (batch, timesteps, 1)
        attention_weights = Softmax(axis=1)(score)   # softmax over timesteps
        weighted = Multiply()([inputs, attention_weights])
        return weighted

    def build_model(self):        
        try:

            inputs = Input(shape=self.input_shape)

            # Multi-scale convolutional blocks
            conv3 = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inputs)
            conv5 = Conv1D(32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inputs)
            conv = Add()([conv3, conv5])
            conv = LayerNormalization()(conv)
            conv = Dropout(0.15)(conv)

            # Residual convolutional block
            conv_res = Conv1D(16, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv)
            conv_res = Add()([conv_res, conv])
            conv_res = LayerNormalization()(conv_res)
            conv_res = Dropout(0.15)(conv_res)

            # Recurrent layers
            rnn = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(1e-4), recurrent_dropout=0.2))(conv_res)
            rnn = GRU(16, return_sequences=True, kernel_regularizer=l2(1e-4), recurrent_dropout=0.2)(rnn)

            # Attention
            rnn = self.attention_layer(rnn)
            rnn = Lambda(lambda x: K.sum(x, axis=1))(rnn)  # Flatten sequence via summation

            if self.exo_dim:
                exo_input = Input(shape=(self.exo_dim,))
                combined = Concatenate()([rnn, exo_input])
                inputs_list = [inputs, exo_input]
            else:
                combined = rnn
                inputs_list = [inputs]

            dense = Dense(16, activation='relu', kernel_regularizer=l2(1e-4))(combined)
            dense = Dropout(0.2)(dense)
            output = Dense(self.output_shape[0], activation='linear')(dense)

            return Model(inputs=inputs_list, outputs=output)

        except Exception as e:
            logger.exception(f'Error building {self.__class__.__name__}: {e}')
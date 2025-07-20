from src.config.config import logger
from abc import ABC, abstractmethod

from tensorflow.keras.layers import Input, Conv1D, LayerNormalization, Dropout, \
    Bidirectional, LSTM, TimeDistributed, GRU, Dense, Add, Multiply, Lambda, Concatenate, Softmax, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model


class IModelBuilder(ABC):
    @abstractmethod
    def build_model(self):
        raise NotImplementedError("Implement in subclass")
    
class RegressionRobustModelBuilder(IModelBuilder):

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

                GRU(self.gru_units, return_sequences=True, kernel_regularizer=l2(self.l2_rate),recurrent_dropout=self.dropout_rate),
                Dropout(self.dropout_rate),

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
            logger.error(f'Error building {self.__class__.__name__}: {e}')


class RegressionSimpleModelBuilder(IModelBuilder):

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
            logger.error(f'Error building {self.__class__.__name__}: {e}')
            raise


class RegressionMultiLayersModelBuilder(IModelBuilder):
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


class LoadKerasModelBuilder(IModelBuilder):
    def __init__(self, model_path):
        self.model_path = model_path

    def build_model(self):
        try:
            return load_model(self.model_path)
        except Exception as e:
            logger.error(f'Error loading model from {self.model_path}: {e}')
            raise
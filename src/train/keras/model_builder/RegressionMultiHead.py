from config import logger
from train.keras.interfaces import IModelBuilder

from tensorflow.keras.layers import Input, Conv1D, Dropout, LSTM, GRU, Dense, \
    LeakyReLU, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class RegressionMultiHead(IModelBuilder):

    def __init__(self, input_shape, output_shape, kernel_rate=5, l2_rate=1e-4, dropout_rate=0.2,
                 conv1D_units=128, gru_units=64, mid_dense_units=32, negative_slope_rate=0.1,
                 attention_heads=4, key_dim=16, low_dense_units=16):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_rate = kernel_rate
        self.l2_rate = l2_rate
        self.dropout_rate = dropout_rate
        self.conv1D_units = conv1D_units
        self.gru_units = gru_units
        self.mid_dense_units = mid_dense_units
        self.negative_slope_rate = negative_slope_rate
        self.attention_heads = attention_heads
        self.key_dim = key_dim
        self.low_dense_units = low_dense_units

    def build_model(self):
        try:
            inp = Input(shape=self.input_shape)

            x = Conv1D(self.conv1D_units, kernel_size=self.kernel_rate, activation='relu',
                       padding='causal', kernel_regularizer=l2(self.l2_rate))(inp)
            x = Dropout(self.dropout_rate)(x)

            x = GRU(self.gru_units, return_sequences=True,
                    kernel_regularizer=l2(self.l2_rate),
                    recurrent_dropout=self.dropout_rate)(x)
            x = Dropout(self.dropout_rate)(x)
            x = LayerNormalization()(x)

            # Multi-Head Attention
            attn_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.key_dim,
                dropout=self.dropout_rate
            )(x, x)

            x = LayerNormalization()(attn_output + x)  # Skip connection
            x = Dropout(self.dropout_rate)(x)

            x = GlobalAveragePooling1D()(x)

            x = Dense(self.mid_dense_units, kernel_regularizer=l2(self.l2_rate))(x)
            x = LeakyReLU(negative_slope=self.negative_slope_rate)(x)
            x = Dropout(self.dropout_rate)(x)

            x = Dense(self.low_dense_units, activation='relu', kernel_regularizer=l2(self.l2_rate))(x)
            x = Dropout(self.dropout_rate)(x)

            out = Dense(self.output_shape[0], activation='linear')(x)

            return Model(inputs=inp, outputs=out)

        except Exception as e:
            logger.error(f'Error building {self.__class__.__name__}: {e}')

from src.config import logger
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, GRU, Dropout, Dense, LayerNormalization, Bidirectional
from tensorflow.keras.regularizers import l2

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

                # 1. Camada Conv1D para captar padrões locais temporais
                Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal'),

                # 2. Camada Bidirectional LSTM para dependências temporais passadas e futuras
                Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))),

                # 3. Camada LayerNormalization para estabilizar treino
                LayerNormalization(),

                Dropout(0.3),

                # 4. Camada GRU para complementar LSTM
                GRU(32, return_sequences=False, kernel_regularizer=l2(1e-4)),

                Dropout(0.2),

                # 5. Camada Dense intermediária
                Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),

                Dropout(0.1),

                # 6. Saída linear para regressão
                Dense(self.output_shape[0], activation='linear')
            ])
        except Exception as e:
            logger.error(f'Error building {self.__class__.__name__}: {e}')
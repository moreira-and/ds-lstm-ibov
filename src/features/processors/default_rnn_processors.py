from config import logger
from ..interfaces import (IPreprocessorStrategy, ITransformStrategy, IGeneratorStrategy, IPostprocessorStrategy)

import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DefaultRnnPreprocessor(IPreprocessorStrategy):

    def __init__(self, transformer: ITransformStrategy, generator: IGeneratorStrategy):
        self._transformer = transformer
        self._generator = generator

    def transform(self, X, y=None):
        X_transformed,y_transformed = self._transformer.transform(X, y)
        return self._generator.generate(X_transformed)


class DefaultRnnTransformStrategy(ITransformStrategy):
    def __init__(
        self, 
        numeric_transformer=RobustScaler(),
        categorical_transformer=OneHotEncoder(sparse_output=False, handle_unknown='ignore')):
            
        self._numeric_transformer = numeric_transformer
        self._categorical_transformer = categorical_transformer

        # Define a ColumnTransformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('numeric', self._numeric_transformer, make_column_selector(dtype_include="number")),
                ('categorical', self._categorical_transformer, make_column_selector(dtype_include="object"))
                ],
            remainder='passthrough'
        )
        

    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.column_transformer.transform(X), y
    
    def get_postprocessor(self, y_train) -> IPostprocessorStrategy:
        return DefaultRnnPostprocessor(self._numeric_transformer,y_train=y_train)
    
    def get_feature_names(self, input_features = None):
        return self.column_transformer.get_feature_names_out(input_features=input_features)


class DefaultRnnGenerator(IGeneratorStrategy):
    def __init__(self, sequence_length=7, batch_size=1):
        self._sequence_length = sequence_length
        self._batch_size = batch_size

    def generate(self, data, targets=None):

        logger.info("Generating Timeseries from dataset...")

        y = targets if targets is not None and len(targets) > 0 else data

        return TimeseriesGenerator(
            data = data,
            targets = y,
            length=self._sequence_length,
            batch_size=self._batch_size,
            shuffle=False
        )
    

class DefaultRnnPostprocessor(IPostprocessorStrategy):

    def __init__(self, transformer,y_train):
        self.__transformer = transformer.fit(y_train)


    def inverse_transform(self, y_predicted):
        # Realiza o inverse_transform normalmente
        y_inversed = self.__transformer.inverse_transform(y_predicted)

        # Verifica se há nomes de colunas disponíveis
        if hasattr(self.__transformer, 'feature_names_in_'):
            column_names = self.__transformer.feature_names_in_
        else:
            # Caso não tenha sido fit com DataFrame, usa nomes genéricos
            column_names = [f'feature_{i}' for i in range(y_inversed.shape[1])]

        return pd.DataFrame(y_inversed, columns=column_names)
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


from src.config import PROCESSED_DATA_DIR

from abc import ABC, abstractmethod


class ModelHandler(ABC):
    @abstractmethod
    def prepare_data(self):
        pass


class DatasetSplitter:
    def __init__(self, train_size_ratio=0.9):
        self.train_size_ratio = train_size_ratio

    def split(self, dataset):
        train_size = int(len(dataset) * self.train_size_ratio)
        return dataset[:train_size], dataset[train_size:]


class DataPreprocessor:
    def __init__(self):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('scaler', StandardScaler(), make_column_selector(dtype_include="number")),
                    ('encoder', OneHotEncoder(), make_column_selector(dtype_include="object"))
                ],
                remainder='passthrough'
            ))
        ])

    def fit_transform(self, data):
        return self.pipeline.fit_transform(data)

    def transform(self, data):
        return self.pipeline.transform(data)


class TimeSeriesDataGenerator:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def generate(self, data):
        generator = TimeseriesGenerator(
            data, data,
            length=self.sequence_length,
            batch_size=1
        )
        return generator


class LstmHandler(ModelHandler):
    def __init__(self, dataset, sequence_length=0.9):

        self.dataset = dataset
        self.sequence_length_ratio = sequence_length

        self.splitter = DatasetSplitter(train_size_ratio=sequence_length)
        self.preprocessor = DataPreprocessor()
        self.sequence_length = int(len(dataset) * (1 - sequence_length))  # Ajuste fino
        self.generator_builder = TimeSeriesDataGenerator(self.sequence_length)

    def prepare_data(self):
        train_data, test_data = self.splitter.split(self.dataset)

        train_X = self.preprocessor.fit_transform(train_data)
        test_X = self.preprocessor.transform(test_data)

        X_all = np.concatenate([train_X, test_X])

        generator = self.generator_builder.generate(X_all)

        n_total = len(X_all)
        n_test = len(test_X)

        self.X_train = np.array([generator[i][0][0] for i in range(n_total - n_test)])
        self.y_train = np.array([generator[i][1][0] for i in range(n_total - n_test)])

        self.X_test = np.array([generator[i][0][0] for i in range(n_total - (n_test+1), n_total)])
        self.y_test = np.array([generator[i][1][0] for i in range(n_total - (n_test+1), n_total)])

    def transform(self, dataset):
        transformed_dataset = self.preprocessor(dataset) 
        generator = self.generator_builder.generate(transformed_dataset)
        return np.array(generator[:][0][0]), np.array(generator[:][1][0])

from typing import Any

import numpy as np

from src.utils.features.splitter_strategy import SplitterStrategy
from src.utils.features.transform_strategy import TransformStrategy
from src.utils.features.generator_strategy import GeneratorStrategy
from src.utils.features.preprocessor_strategy import DefaultLstmPreprocessor

from abc import ABC, abstractmethod


class PrepareDataTemplate(ABC):
    @abstractmethod
    def prepare_data(self):
        pass
    
    @abstractmethod
    def get_preprocessor(self): 
        pass

    @abstractmethod
    def get_postprocessor(self): 
        pass


class DefaultLstmPrepareDataTemplate(PrepareDataTemplate):
    def __init__(self, dataset, splitter: SplitterStrategy, transformer: TransformStrategy, generator: GeneratorStrategy):
        self.dataset = dataset
        self.splitter = splitter
        self.transformer = transformer
        self.generator = generator

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def prepare_data(self):
        train_data, test_data = self.splitter.split(X = self.dataset)

        train_X = self.transformer.fit_transform(X = train_data) # train fit
        test_X = self.transformer.transform(X = test_data) # test transform with train fit (real)

        X_all = np.concatenate([train_X, test_X]) # concat to generate

        generator = self.generator.generate(X_all)

        n_total = len(generator)
        n_test = len(test_data)

        self._X_train = np.array([generator[i][0][0] for i in range(n_total - n_test)])
        self._y_train = np.array([generator[i][1][0] for i in range(n_total - n_test)])

        self._X_test = np.array([generator[i][0][0] for i in range(n_total - (n_test+1), n_total)])
        self._y_test = np.array([generator[i][1][0] for i in range(n_total - (n_test+1), n_total)])

    def get_data(self):
        return self._X_train,self._X_test,self._y_train,self._y_test   

    def get_preprocessor(self):
        return DefaultLstmPreprocessor(self.transformer, self.generator)
    
    def get_postprocessor(self):
        return self.transformer.get_postprocessor()
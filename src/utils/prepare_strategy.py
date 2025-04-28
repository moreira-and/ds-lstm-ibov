from typing import Any

import numpy as np

from src.utils.splitter_strategy import SplitterStrategy
from src.utils.preprocessor_strategy import PreprocessorStrategy
from src.utils.generator_strategy import GeneratorBuilder

from abc import ABC, abstractmethod


class PrepareDataTemplate(ABC):
    @abstractmethod
    def prepare_data(self):
        pass


class LstmPrepareTemplate(PrepareDataTemplate):
    def __init__(self, dataset, splitter: SplitterStrategy, preprocessor: PreprocessorStrategy, generator: GeneratorBuilder):
        self.dataset = dataset
        self.splitter = splitter
        self.preprocessor = preprocessor
        self.generator_builder = generator

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        train_data, test_data = self.splitter.split(self.dataset)

        train_X = self.preprocessor.fit_transform(train_data) # train fit
        test_X = self.preprocessor.transform(test_data) # test transform with train fit (real)

        X_all = np.concatenate([train_X, test_X]) # concat to generate

        generator = self.generator_builder.generate(X_all)

        n_train = len(train_X)
        n_test = len(test_X)

        self.X_train = np.array([generator[i][0][0] for i in range(n_train)])
        self.y_train = np.array([generator[i][1][0] for i in range(n_train)])

        self.X_test = np.array([generator[i][0][0] for i in range(n_train, n_train+n_test)])
        self.y_test = np.array([generator[i][1][0] for i in range(n_train, n_train+n_test)])
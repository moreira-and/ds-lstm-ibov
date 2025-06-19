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
    def __init__(self, dataset,targets, splitter: SplitterStrategy, transformer: TransformStrategy, generator: GeneratorStrategy):
        self.dataset = dataset
        self.targets = targets
        self.splitter = splitter
        self.transformer = transformer
        self.generator = generator

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def prepare_data(self):
        train_data, test_data = self.splitter.split(X = self.dataset)

        train_X,train_y = self.transformer.fit_transform(X = train_data) # train fit
        
         # initialize empty array for test data

        if len(test_data) == 0 or test_data is None:
            # if no test data, use train data for generating features
            test_X = None
            X_all = train_X # concat to generate
        else:
            test_X = self.transformer.transform(X = test_data)
            X_all = np.concatenate([train_X, test_X]) # concat to generate

        if self.targets is None or len(self.targets) == 0:
            # if no targets, generate features for all data
            generator = self.generator.generate(data=X_all)
        else:    
            feature_names = self.transformer.get_feature_names() # get feature names]
            y_features = [
                (i, feature)
                for i, feature in enumerate(feature_names)
                if any(t in feature for t in self.targets)
            ]
            y_index = [i for i, _ in y_features]  
            generator = self.generator.generate(data=X_all,targets = X_all[:,y_index])   

        n_test = len(test_data)
        n_total = len(generator)            
        train_end = n_total - n_test
        
        self._X_train = np.array([generator[i][0][0] for i in range(train_end)])
        self._y_train = np.array([generator[i][1][0] for i in range(train_end)])
        self._X_test = np.array([generator[i][0][0] for i in range(train_end, n_total)])
        self._y_test = np.array([generator[i][1][0] for i in range(train_end, n_total)])

    def get_data(self):
        return self._X_train,self._X_test,self._y_train,self._y_test   

    def get_preprocessor(self):
        return DefaultLstmPreprocessor(self.transformer, self.generator)
    
    def get_postprocessor(self):

        X,_ = self.splitter.split(X = self.dataset)

        if self.targets is None or len(self.targets) == 0:
            # if no targets, return all columns
            filtered_columns = X.columns.tolist()
        else:
            filtered_columns = [
                feature for feature in X
                if any(t in feature for t in self.targets)
            ]

        return self.transformer.get_postprocessor(X.loc[:, filtered_columns])
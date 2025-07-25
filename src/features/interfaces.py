from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import pandas as pd
import numpy as np
# Abstract base class for data cleaning handlers
class ICleanStrategy(ABC):
    @abstractmethod
    def clear(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None
             ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Abstract method that should be implemented by any class that inherits from CleanHandler.
        This method should be used for cleaning the data (X) and optionally the target (y).
        """
        pass

class ISelectStrategy(ABC):
    @abstractmethod
    def select(self,X,y=None):
        pass

class IGeneratorStrategy(ABC):
    @abstractmethod
    def generate(self, data,targets=None):
        pass

class IPreprocessorStrategy(ABC):
    @abstractmethod
    def transform(self, X, y=None):
        pass

class IPostprocessorStrategy(ABC):  
    
    @abstractmethod
    def inverse_transform(self, y_predicted):
        pass
    
    def transform(self, y_predicted):
        return self.inverse_transform(y_predicted)
    

class ITransformStrategy(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @abstractmethod
    def get_postprocessor(self,y_train) -> IPostprocessorStrategy:
        pass
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    

class IPrepareDataTemplate(ABC):
    @abstractmethod
    def prepare_data(self):
        pass
    
    @abstractmethod
    def get_preprocessor(self): 
        pass

    @abstractmethod
    def get_postprocessor(self): 
        pass
    
class ISplitterStrategy(ABC):
    @abstractmethod
    def split(self,X, y=None):
        pass
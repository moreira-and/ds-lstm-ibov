from config import logger
from .interfaces import IPreprocessorStrategy

from typing import List,Optional, Tuple, Union
import pandas as pd
import numpy as np

class PreprocessorPipeline(IPreprocessorStrategy):
    def __init__(self, preprocessors:List[IPreprocessorStrategy]):
        self.preprocessors = preprocessors
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """
        Fits the transformation strategy on the dataset.
        """
        pass

    
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Applies the transformation to the dataset.
        """
        pass

    
    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Applies the transformation to the dataset.
        """
        pass
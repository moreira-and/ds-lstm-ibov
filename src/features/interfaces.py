"""
Interface module defining abstract base classes for each stage in a data processing pipeline.

Each interface represents a single responsibility such as data cleaning, selection, transformation, and postprocessing.
These should be implemented by concrete strategy classes adhering to the defined signatures, enabling modular,
extensible, and testable pipelines.

Author: [Your Name]
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
import pandas as pd
import numpy as np


class ICleanStrategy(ABC):
    @abstractmethod
    def clear(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Defines an interface for cleaning raw datasets.

        Parameters:
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix to be cleaned.
        y : pd.Series | np.ndarray | None, default=None
            Optional target variable.

        Returns:
        -------
        Tuple containing cleaned X and y.
        """
        pass


class ISelectStrategy(ABC):
    @abstractmethod
    def select(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Defines an interface for selecting relevant features or samples from a dataset.
        """
        pass


class IGeneratorStrategy(ABC):
    @abstractmethod
    def generate(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        targets: Optional[list[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Defines an interface for generating new data or features.
        """
        pass


class IPreprocessorStrategy(ABC):
    @abstractmethod
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Defines an interface for transforming raw data before modeling.
        """
        pass


class IPostprocessorStrategy(ABC):
    @abstractmethod
    def inverse_transform(
        self,
        y_predicted: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Defines an interface to reverse transformations applied during preprocessing.
        """
        pass

    def transform(
        self,
        y_predicted: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        return self.inverse_transform(y_predicted)


class ITransformStrategy(ABC):
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """
        Fits the transformation strategy on the dataset.
        """
        pass

    @abstractmethod
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Applies the transformation to the dataset.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Returns the list of transformed feature names.
        """
        pass

    @abstractmethod
    def get_postprocessor(
        self,
        y_train: Union[pd.Series, np.ndarray]
    ) -> IPostprocessorStrategy:
        """
        Returns the corresponding postprocessor for inverse transformations.
        """
        pass

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        self.fit(X, y)
        return self.transform(X, y)


class IPrepareDataTemplate(ABC):
    @abstractmethod
    def prepare_data(self) -> Tuple[Any, Any]:
        """
        Template for data preparation logic.
        """
        pass

    @abstractmethod
    def get_preprocessor(self) -> IPreprocessorStrategy:
        pass

    @abstractmethod
    def get_postprocessor(self) -> IPostprocessorStrategy:
        pass


class ISplitterStrategy(ABC):
    @abstractmethod
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple:
        """
        Defines an interface to split data into train/test sets or time-based folds.
        """
        pass

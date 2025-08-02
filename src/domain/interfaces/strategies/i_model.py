from domain.entities.data.model_input_data import ModelInputData
from domain.entities.data.model_output_data import ModelOutputData

from domain.entities.metadata.model_metadata import ModelMetadata
from abc import ABC, abstractmethod

class IModel(ABC):
    """
    This interface is responsible for model training and prediction.

    It is used to standardize orchestration in the application layer and
    to facilitate implementation in the infrastructure layer, abstracting
    the underlying libraries and frameworks involved.
    """

    @abstractmethod
    def train(self, data: ModelInputData) -> ModelMetadata:
        pass

    @abstractmethod
    def predict(self, data: ModelInputData) -> ModelOutputData:
        pass

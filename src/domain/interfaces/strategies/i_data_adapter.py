from src.domain.entities.data.selected_data import SelectedData
from domain.entities.data.model_input_data import ModelInputData
from domain.entities.data.model_output_data import ModelOutputData
from domain.entities.data.predicted_data import PredictedData

from domain.entities.metadata.adapter_metadata import AdapterMetadata


from abc import ABC, abstractmethod

class IDataAdapter(ABC):
    """
    This interface handles data processing to adapt data for model training
    and, after training, returns the predicted data scaled back to the original range.

    It is used to standardize orchestration in the application layer and
    to facilitate implementation in the infrastructure layer, abstracting
    away the libraries and frameworks involved.
    """

    @abstractmethod
    def fit(self, data: SelectedData) -> AdapterMetadata:
        pass

    @abstractmethod
    def transform(self, data: SelectedData) -> ModelInputData:
        pass

    @abstractmethod
    def inverse_transform(self, data: ModelOutputData) -> PredictedData:
        pass

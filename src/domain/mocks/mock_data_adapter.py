from src.domain.entities.data.selected_data import SelectedData
from src.domain.entities.data.model_input_data import ModelInputData
from src.domain.entities.data.model_output_data import ModelOutputData
from src.domain.entities.data.predicted_data import PredictedData
from src.domain.entities.metadata.adapter_metadata import AdapterMetadata
from src.domain.interfaces.strategies.i_data_adapter import IDataAdapter


class MockDataSelector(IDataAdapter):
    """
    A jumper/mock implementation of IDataAdapter.
    Useful for testing, composition, or bypassing selection logic in pipelines.
    """

    def fit(self, data: SelectedData) -> AdapterMetadata:
        # Mock metadata output
        metadata = {"status": "pass", "note": "MockDataSelector used"}
        return AdapterMetadata(metadata=metadata)

    def transform(self, data: SelectedData) -> ModelInputData:
        return ModelInputData(data.data)

    def inverse_transform(self, data: ModelOutputData) -> PredictedData:
        return PredictedData(data.data)
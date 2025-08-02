from src.domain.entities.data.model_input_data import ModelInputData
from src.domain.entities.data.model_output_data import ModelOutputData
from domain.entities.metadata.model_metadata import ModelMetadata
from src.domain.interfaces.strategies.i_model import IModel


class MockModel(IModel):
    """
    A jumper/mock implementation of IModel.
    Useful for testing, composition, or bypassing selection logic in pipelines.
    """

    def train(self, data: ModelInputData) -> ModelMetadata:
        # Mock metadata output
        metadata = {"status": "pass", "note": "MockModel used"}
        return ModelMetadata(metadata=metadata)

    def predict(self, data: ModelInputData) -> ModelOutputData:
        return ModelOutputData(data.data)

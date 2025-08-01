from src.domain.entities.dto.raw_data_dto import RawDataDto
from src.domain.entities.dto.predicted_data_dto import PredictedDataDto
from src.domain.interfaces.i_model import IModel
from src.domain.interfaces.i_preprocessor import IPreProcessor
from src.domain.interfaces.i_postprocessor import IPostProcessor

class PredictUseCase:
    def __init__(
        self,
        preprocessor: IPreProcessor,
        model: IModel,
        postprocessor: IPostProcessor
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    def execute(self, raw_data: RawDataDto) -> PredictedDataDto:
        processed = self.preprocessor.transform(raw_data)
        prediction = self.model.predict(processed)
        result = self.postprocessor.inverse_transform(prediction)
        return result

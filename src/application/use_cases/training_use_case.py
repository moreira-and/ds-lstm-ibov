from src.domain.entities.dto.raw_data_dto import RawDataDto
from src.domain.entities.dto.training_result_dto import TrainingResultDto
from src.domain.interfaces.i_model import IModel
from src.domain.interfaces.i_preprocessor import IPreProcessor

class TrainUseCase:
    def __init__(
        self,
        preprocessor: IPreProcessor,
        model: IModel,
    ):
        self.preprocessor = preprocessor
        self.model = model

    def execute(self, raw_data: RawDataDto) -> TrainingResultDto:
        processed_data = self.preprocessor.transform(raw_data)
        training_metrics = self.model.train(processed_data)
        return TrainingResultDto(metrics=training_metrics)

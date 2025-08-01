from src.domain.entities.dto.raw_data_dto import RawDataDto
from src.domain.entities.dto.tuning_result_dto import TuningResultDto
from src.domain.interfaces.i_model import ITunableModel
from src.domain.interfaces.i_preprocessor import IPreProcessor

class TuneUseCase:
    def __init__(
        self,
        preprocessor: IPreProcessor,
        model: ITunableModel,
    ):
        self.preprocessor = preprocessor
        self.model = model

    def execute(self, raw_data: RawDataDto, labels: RawDataDto, param_grid: dict) -> TuningResultDto:
        processed_data = self.preprocessor.transform(raw_data)
        tuning_metrics = self.model.tune(processed_data, labels, param_grid)
        return TuningResultDto(metrics=tuning_metrics)

from src.domain.entities.dto.raw_data_dto import RawDataDto
from domain.interfaces.i_data_adapter import IPreProcessor
from src.domain.interfaces.i_postprocessor import IPostProcessor

from typing import Tuple

class ProcessDataUseCase:
    def __init__(
        self,
        preprocessor: IPreProcessor,
        postprocessor: IPostProcessor
    ):
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def execute(self, data: RawDataDto) -> Tuple:
        #processed = self.preprocessor.fit(raw_data)
        #result = self.postprocessor.inverse_transform(processed)
        return self.preprocessor, self.preprocessor

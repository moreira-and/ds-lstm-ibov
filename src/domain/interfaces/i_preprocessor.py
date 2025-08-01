from src.domain.entities.dto.raw_data_dto import RawDataDto
from src.domain.entities.dto.fitting_result_dto import FittingResultDto
from src.domain.entities.dto.transformed_data_dto import TransformedDataDto

from typing import Any
from abc import ABC, abstractmethod

class IPreProcessor(ABC):

    @abstractmethod
    def fit(self, data: RawDataDto) -> FittingResultDto:
        pass

    @abstractmethod
    def transform(self, data: RawDataDto) -> TransformedDataDto:
        pass


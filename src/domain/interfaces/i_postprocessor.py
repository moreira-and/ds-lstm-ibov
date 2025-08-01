from domain.entities.dto.transformed_data_dto import TransformedDataDto
from domain.entities.dto.predicted_data_dto import PredictedDataDto

from typing import Any
from abc import ABC, abstractmethod

class IPostProcessor(ABC):

    @abstractmethod
    def inverse_transform(self, data: TransformedDataDto) -> PredictedDataDto:
        pass


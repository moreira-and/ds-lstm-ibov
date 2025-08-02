from domain.entities.data.raw_data import RawData
from domain.entities.data.predicted_data import PredictedData

from abc import ABC, abstractmethod

class IPredictorFullCycle(ABC):
    @abstractmethod
    def execute(self, input_data: RawData) -> PredictedData:
        pass

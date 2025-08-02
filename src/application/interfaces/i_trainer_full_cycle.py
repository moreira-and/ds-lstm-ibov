from domain.entities.data.raw_data import RawData
from domain.entities.metadata.full_cycle_metadata import FullCycleMetadata

from abc import ABC, abstractmethod


class ITrainerFullCycle(ABC):
    @abstractmethod
    def execute(self, input_data: RawData) -> FullCycleMetadata:
        pass
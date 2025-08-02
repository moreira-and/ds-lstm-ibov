from abc import ABC, abstractmethod
from domain.entities.metadata.model_metadata import ModelMetadata

class ModelValidator(ABC):
    @abstractmethod
    def is_valid(self, metadata: ModelMetadata) -> bool:
        """
        Determines if the trained model meets the acceptance criteria.
        """
        pass

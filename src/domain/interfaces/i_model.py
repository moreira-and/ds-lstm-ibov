from abc import ABC, abstractmethod
from typing import Any

class IModel(ABC):
    @abstractmethod
    def train(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        pass

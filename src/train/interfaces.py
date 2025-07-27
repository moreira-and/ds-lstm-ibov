from abc import ABC, abstractmethod
from typing import Any

class ITrainerRunner(ABC):
    @abstractmethod
    def train(self, X_train, y_train) -> Any:
        pass
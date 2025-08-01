from abc import ABC, abstractmethod
from typing import Optional, List

class IModelBuilder(ABC):
    @abstractmethod
    def build_model(self):
        pass
        targets: Optional[List[str]] = None
    
from abc import ABC, abstractmethod
from typing import List
from src.domain.interfaces.strategies.i_model import IModel


class IModelQueryRepository(ABC):
    @abstractmethod
    def get_by_id(self, model_id: str) -> IModel:
        pass

    @abstractmethod
    def list_all(self) -> List[IModel]:
        pass

    @abstractmethod
    def find_by_status(self, status: str) -> List[IModel]:
        pass

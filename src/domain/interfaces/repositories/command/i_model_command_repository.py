from abc import ABC, abstractmethod
from src.domain.interfaces.strategies.i_model import IModel


class IModelCommandRepository(ABC):
    @abstractmethod
    def save(self, model: IModel) -> None:
        pass

    @abstractmethod
    def delete(self, model_id: str) -> None:
        pass

    @abstractmethod
    def update_status(self, model_id: str, status: str) -> None:
        pass

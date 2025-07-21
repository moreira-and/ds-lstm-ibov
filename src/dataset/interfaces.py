from abc import ABC, abstractmethod

class IDatasetLoader(ABC):
    @abstractmethod
    def load(self):
        pass
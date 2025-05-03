from abc import ABC, abstractmethod

class ExperimentBuilder(ABC):
    @abstractmethod
    def run(self):
        pass

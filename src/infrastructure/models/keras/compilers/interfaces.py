from abc import ABC, abstractmethod

class ICompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass
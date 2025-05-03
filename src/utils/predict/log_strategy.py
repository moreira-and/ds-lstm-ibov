from abc import ABC, abstractmethod

class LogStrategy(ABC):
    @abstractmethod
    def log(self):
        pass


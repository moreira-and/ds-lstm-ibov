from abc import ABC, abstractmethod

class ITuneRunner(ABC):
    @abstractmethod
    def run(self, X_train, y_train):
        pass
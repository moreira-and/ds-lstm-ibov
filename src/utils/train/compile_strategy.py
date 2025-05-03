from src.config import logger

from abc import ABC, abstractmethod


class CompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class ClassificationCompileStrategy(CompileStrategy):
    def __init__(self, optimizer = None, loss = None, metrics = None):
        self.optimizer = optimizer or 'adam'
        self.loss = loss or 'mse'
        self.metrics = metrics or ['accuracy']

    def compile(self, model):
        try:
            model.compile(
                optimizer = self.optimizer,
                loss = self.loss,
                metrics = self.metrics
            )
        except Exception as e:
            logger.error(f'Error compilling {self.__class__.__name__}: {e}')

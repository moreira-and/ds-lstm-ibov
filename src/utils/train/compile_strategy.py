from src.config import logger

from abc import ABC, abstractmethod


class CompileStrategy(ABC):
    @abstractmethod
    def compile(self, model):
        pass

class ClassificationCompileStrategy(CompileStrategy):
    def __init__(self, optimizer = 'adam', loss = 'mse', metrics = ['accuracy']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compile(self, model):
        try:
            model.compile(
                optimizer = self.optimizer,
                loss = self.loss,
                metrics = self.metrics
            )
        except Exception as e:
            logger.error(f'Error compilling {self.__class__.__name__}: {e}')

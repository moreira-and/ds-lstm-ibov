from abc import ABC, abstractmethod

from src.config import logger

from src.utils.train.model_builder import ModelBuilder
from src.utils.train.compile_strategy import CompileStrategy
from src.utils.train.train_strategy import TrainStrategy

class ModelTemplate(ABC):
    @abstractmethod
    def run(self, X_train, y_train):
        pass


class ModelKerasPipeline(ModelTemplate):
    def __init__(self, model_builder: ModelBuilder, compiler: CompileStrategy, trainer: TrainStrategy):
        self.model_builder = model_builder
        self.compiler = compiler
        self.trainer = trainer
        self.model = None

    def run(self, X_train, y_train):
        try:
            self.model = self.model_builder.build_model()
            self.compiler.compile(self.model)
            history = self.trainer.train(self.model, X_train, y_train)
            return self.model, history
        except Exception as e:
            logger.error(f'Error running {self.__class__.__name__}: {e}')





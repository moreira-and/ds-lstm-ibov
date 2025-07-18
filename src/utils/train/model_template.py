from abc import ABC, abstractmethod

from src.config import logger

from src.utils.train.model_builder import IModelBuilder
from src.utils.train.compile_strategy import ICompileStrategy
from src.utils.train.train_strategy import ITrainStrategy

class IModelTemplate(ABC):
    @abstractmethod
    def run(self, X_train, y_train):
        raise NotImplementedError("Implement in subclass")


class ModelKerasPipeline(IModelTemplate):
    def __init__(self, model_builder: IModelBuilder, compiler: ICompileStrategy, trainer: ITrainStrategy):
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





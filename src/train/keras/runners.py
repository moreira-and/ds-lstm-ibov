from config import logger

from ..interfaces import ITrainerRunner

from train.keras.interfaces import IGenerator
from train.keras.interfaces import IModelBuilder
from train.keras.interfaces import ICompileStrategy
from train.keras.interfaces import ITrainStrategy


class TrainerKerasRunner(ITrainerRunner):
    def __init__(self, generator: IGenerator ,model_builder: IModelBuilder, compiler: ICompileStrategy, trainer: ITrainStrategy):
        self.generator: generator
        self.model_builder = model_builder
        self.compiler = compiler
        self.trainer = trainer
        self.model = None

    def train(self, X_train, y_train):
        try:
            self.model = self.model_builder.build_model()
            self.compiler.compile(self.model)
            history = self.trainer.train(self.model, X_train, y_train)
            return self.model, history
        except Exception as e:
            logger.error(f'Error running {self.__class__.__name__}: {e}')
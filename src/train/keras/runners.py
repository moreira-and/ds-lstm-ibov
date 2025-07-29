from config import logger
from ..interfaces import ITrainerRunner
from .model_builders.interfaces import IModelBuilder
from .compilers.interfaces import ICompileStrategy
from .trainers.interfaces import ITrainStrategy
from .trainers.generators import sliding_window_generator

import pandas as pd
from typing import Tuple, Optional
from tensorflow.keras.models import Model


class TrainerKerasRunner(ITrainerRunner):
    def __init__(
        self,
        model_builder: IModelBuilder,
        compiler: ICompileStrategy,
        trainer: ITrainStrategy,
    ):
        self.model_builder = model_builder
        self.compiler = compiler
        self.trainer = trainer
        self.model: Optional[Model] = None

    def train(self, df: pd.DataFrame) -> Tuple[Model, dict]:
        if self.model is None:
            self.model = self.model_builder.build_model()

        try:
            self.compiler.compile(self.model)
            history = self.trainer.train(self.model, df)

            if hasattr(history, "history") and isinstance(history.history, dict):
                return self.model, history.history
            else:
                raise TypeError("Expected Keras History object with `.history` attribute")

        except Exception as e:
            logger.error(f"Error during training in {self.__class__.__name__}: {e}")
            raise

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError(f"Model has not been trained yet in {self.__class__.__name__}")
        
        return self.model.predict(sliding_window_generator(X))

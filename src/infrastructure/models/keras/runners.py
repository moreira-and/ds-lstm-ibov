from config import logger
from ..interfaces import ITrainerRunner
from .model_builders.interfaces import IModelBuilder
from .compilers.interfaces import ICompileStrategy
from .trainers.interfaces import ITrainStrategy

import pandas as pd
from typing import Tuple, Optional
from tensorflow.keras.models import Model


class TrainerKerasRunner(IModelTemplate):
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
        try:
            if self.model is None:
                self.model = self.model_builder.build_model()
        
            self.compiler.compile(self.model)
            history = self.trainer.train(self.model, df)

            if hasattr(history, "history") and isinstance(history.history, dict):
                return self.model, history.history
            else:
                logger.exception("Expected Keras History object with `.history` attribute")
                raise

        except Exception as e:
            logger.exception(f"Error during training in {self.__class__.__name__}: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.model is None:
                logger.error(f"Model has not been trained yet in {self.__class__.__name__}")
                raise
            
            return self.trainer.predict(model=self.model,df=df)
        
        except Exception as e:
            logger.exception(f"Error during predict in {self.__class__.__name__}: {e}")
            raise

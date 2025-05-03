import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import numpy as np


from abc import ABC

from src.config import logger
from src.utils.train.model_builder import ModelBuilder
from src.utils.train.compile_strategy import CompileStrategy
from src.utils.train.train_strategy import TrainStrategy

class ModelTemplate(ABC):
    def run(self, X_train, y_train):
        pass


class ModelKerasPipeline(ModelTemplate):
    def __init__(self, builder: ModelBuilder, compiler: CompileStrategy, trainer: TrainStrategy):
        self.builder = builder
        self.compiler = compiler
        self.trainer = trainer

    def run(self, X_train, y_train):
        try:
            model = self.builder.build_model()
            self.compiler.compile(model)
            history = self.trainer.train(model, X_train, y_train)
            return model, history
        except Exception as e:
            logger.error(f'Error running {self.__class__.__name__}: {e}')

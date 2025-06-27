from abc import ABC, abstractmethod
from src.config import MODELS_DIR, logger

import keras_tuner as kt
from keras_tuner import HyperParameters

from src.utils.train.model_builder import RegressionRobustModelBuilder
from src.utils.train.compile_strategy import CompileStrategy,RegressionCompileStrategy


class TunerBuilder(ABC):
    @abstractmethod
    def _build_model(self,hp: HyperParameters):
        return

    @abstractmethod
    def build_tuner(self):
        return

class RegressionRobustModelTuner(TunerBuilder):

    def __init__(self, input_shape, output_shape,max_trials : int =10,project_name : str = "default", compile_strategy : CompileStrategy = RegressionCompileStrategy()):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_trials = max_trials
        self.project_name = project_name
        self.compile_strategy = compile_strategy

    def _build_model(self,hp: HyperParameters):

        l2_rate = hp.Float("l2_rate", min_value=1e-4, max_value=1e-3, step=1e-4)
        dropout_rate= hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
        conv1D_units = hp.Int("conv1D_units", min_value=64, max_value=256, step=32)
        lstm_units = hp.Int("lstm_units", min_value=32, max_value=128, step=16)
        dense_units = hp.Int("dense_units", min_value=32, max_value=128, step=16)
        gru_units = hp.Int("gru_units", min_value=32, max_value=128, step=16) 

        builder = RegressionRobustModelBuilder(
            input_shape = self.input_shape,
            output_shape = self.output_shape,
            l2_rate = l2_rate,
            dropout_rate = dropout_rate,
            conv1D_units = conv1D_units,
            lstm_units = lstm_units,
            dense_units = dense_units,
            gru_units = gru_units
        )

        model =  builder.build_model()
        self.compile_strategy.compile(model)

        return model

    def build_tuner(self):
        return kt.BayesianOptimization(
            self._build_model,
            objective="val_loss",
            max_trials=self.max_trials ,
            executions_per_trial=1,
            directory=MODELS_DIR,
            project_name=self.project_name
        )
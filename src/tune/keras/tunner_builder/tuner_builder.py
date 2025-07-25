
from src.config.paths import MODELS_DIR
from ..interfaces import ITunerBuilder

from keras_tuner import (HyperParameters,BayesianOptimization)

from tensorflow.keras.losses import Huber

from src.utils.train.model_builder import RegressionRobustModelBuilder
from src.utils.train.compile_strategy import ICompileStrategy,RegressionCompileStrategy



class RegressionRobustModelTuner(ITunerBuilder):

    def __init__(self, input_shape, output_shape,max_trials : int =10, project_name : str = "default_tune"):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_trials = max_trials
        self.project_name = project_name
        self.compile_strategy = None

    def get_model(self,hp: HyperParameters):

        delta_rate = hp.Float("delta_rate", min_value=0.8, max_value=3.0, step=0.1)
        self.compile_strategy= RegressionCompileStrategy(loss = Huber(delta=delta_rate))

        kernel_rate= hp.Int("kernel_rate", min_value=3, max_value=7, step=1)
        l2_rate = hp.Float("l2_rate", min_value=1e-5, max_value=1e-2, sampling="log")
        dropout_rate= hp.Float("dropout_rate", min_value=0.1, max_value=0.5)
        conv1D_units = hp.Int("conv1D_units", min_value=64, max_value=256, step=32)
        gru_units = hp.Int("gru_units", min_value=32, max_value=128, step=8)
        mid_dense_units = hp.Int("mid_dense_units", min_value=32, max_value=128, step=16)
        negative_slope_rate = hp.Float("alpha_rate", min_value=0.01, max_value=0.2, step=0.01)
        lstm_units = hp.Int("lstm_units", min_value=32, max_value=128, step=8)
        low_dense_units = hp.Int("low_dense_units", min_value=32, max_value=128, step=16)

        builder = RegressionRobustModelBuilder(
            input_shape = self.input_shape,
            output_shape = self.output_shape,
            kernel_rate = kernel_rate,
            l2_rate = l2_rate,
            dropout_rate = dropout_rate,
            conv1D_units = conv1D_units,
            gru_units = gru_units,
            mid_dense_units = mid_dense_units,
            negative_slope_rate = negative_slope_rate,
            lstm_units = lstm_units,
            low_dense_units = low_dense_units
        )

        model =  builder.build_model()
        self.compile_strategy.compile(model)

        return model

    def build_tuner(self):
        return BayesianOptimization(
            self.get_model,
            objective="val_loss",
            max_trials=self.max_trials ,
            executions_per_trial=1,
            directory=MODELS_DIR,
            project_name=self.project_name
        )
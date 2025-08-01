from src.log.interfaces import ILogStrategy
from src import config

from mlflow import pyfunc
from mlflow.pyfunc import PythonModelContext, PythonModel

import pickle
from pathlib import Path
import pandas as pd

class PythonModelPipeline(PythonModel):
    def load_context(self, context):
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(context.artifacts["postprocessor"], "rb") as f:
            self.postprocessor = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
        X_transformed = self.preprocessor.transform(model_input)
        y_pred = self.model.predict(X_transformed)
        return self.postprocessor.transform(y_pred)

class LogPythonModel(ILogStrategy):
    def __init__(self, model_name, input_exemple_path:Path = None, prep_dest:Path =None, post_dest:Path =None, model_path:Path=None):
        """
        Initializes the logger with the model and input example.
        """
        self.model_name = model_name
        self.prep_dest = prep_dest or config.MAIN_PREPROCESSOR_FILE
        self.post_dest = post_dest or config.MAIN_MODEL_FILE
        self.model_path = model_path or config.MAIN_POSTPROCESSOR_FILE
        self.input_exemple_path = input_exemple_path or config.TEST_RAW_FILE

    def log(self, **kwargs):
        """
        Logs the Python model to MLflow.
        """
        config.logger.info("Logging Python model...")

        input_example = pd.read_csv(self.input_exemple_path, index_col=0)

        pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=PythonModelPipeline(),
            artifacts={
                "preprocessor": self.prep_dest,
                "postprocessor": self.post_dest,
                "model": self.model_path,
            },
            input_example = input_example,
            registered_model_name= self.model_name
        )
  

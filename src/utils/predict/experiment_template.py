from abc import ABC, abstractmethod
from pathlib import Path

import cloudpickle
import mlflow
from mlflow import log_param, log_metric
from mlflow.types.schema import Schema
from mlflow.models import ModelSignature, ModelInputExample


from src.config import MLFLOW_TRACKING_URI
from src.utils.predict.py_func_model_builder import PyFuncModelTemplate

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



class IExperimentTemplate(ABC):

    @abstractmethod
    def run(self):
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def build_pyfunc_model(self) -> PyFuncModelTemplate:
        """Build and return a Python model."""
        raise NotImplementedError("Implement in subclass")

    def set_tags(self, tags: dict):
        """Set tags for the experiment."""
        self.tags = tags
        mlflow.set_tags(tags)

    def log_param(self, key: str, value: str):
        """Log a parameter to the experiment."""
        log_param(key, value)

    def log_metric(self, key: str, value: float):
        """Log a metric to the experiment."""
        log_metric(key, value)

    def log_model(self, model, artifact_path: str, signature: ModelSignature = None):
        """Log a model to the experiment."""
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=model, signature=signature)

    def log_artifact(self, artifact_path: str, artifact_file: str):
        """Log an artifact to the experiment."""
        mlflow.log_artifact(artifact_path=artifact_file)

    def log_input_example(self, input_example: ModelInputExample):
        """Log an input example to the experiment."""
        mlflow.log_input_example(input_example)

    def log_model_signature(self, signature: ModelSignature):
        """Log a model signature to the experiment."""
        mlflow.log_model_signature(signature)

    def log_model_input_schema(self, input_schema: Schema):
        """Log a model input schema to the experiment."""
        mlflow.log_model_input_schema(input_schema)

    def log_model_output_schema(self, output_schema: Schema):
        """Log a model output schema to the experiment."""
        mlflow.log_model_output_schema(output_schema)

    def log_model_input_example(self, input_example: ModelInputExample):
        """Log a model input example to the experiment."""
        mlflow.log_model_input_example(input_example)


class LSTMRegressionExperiment(IExperimentTemplate):
    def __init__(self, model_path: Path, preprocessor_path: Path, postprocessor_path: Path, X_test, y_test, py_func:PyFuncModelTemplate ,tags: dict = None):
        self.model = cloudpickle.dump(self.model, open(model_path, "wb"))
        self.preprocessor = cloudpickle.dump(self.preprocessor, open(preprocessor_path, "wb"))
        self.postprocessor = cloudpickle.dump(self.postprocessor, open(postprocessor_path, "wb"))
        self.X_test = X_test
        self.y_test = y_test
        self.py_func = py_func
        if tags:
            self.set_tags(tags)

    def run(self):
        """Run the experiment."""

        mlflow.set_experiment("LSTM_Regression_Experiment")
        
        with mlflow.start_run() as run:
            # Log basic model info
            self.log_param("model_type", "LSTM")
            self.log_param("framework", "Keras")
            self.log_param("preprocessing", type(self.preprocessor).__name__)
            self.log_param("postprocessing", type(self.postprocessor).__name__)

            # Predict and compute metrics
            X_proc = self.preprocessor.transform(self.X_test)
            y_pred_raw = self.model.predict(X_proc)
            y_pred = self.postprocessor.inverse_transform(y_pred_raw)
            mse = ((y_pred.flatten() - self.y_test.flatten())**2).mean()

            self.log_metric("mse", float(mse))

            artifacts = {
                "model": self.model_path,
                "preprocessor": self.preprocessor_path,
                "postprocessor": self.postprocessor_path
            }

            mlflow.pyfunc.log_model(
                artifact_path="lstm_pyfunc",
                python_model=self._build_pyfunc_model(),
                artifacts=artifacts,
                input_example=self.X_test[:-2],
                output_example=self.y_test[:-2],
                signature="infer",
                pip_requirements="infer" # pipreqs on the future
            )


    def _build_pyfunc_model(self):
        return self.py_func(
            preprocessor=self.preprocessor, 
            model=self.model, 
            postprocessor=self.postprocessor
            )
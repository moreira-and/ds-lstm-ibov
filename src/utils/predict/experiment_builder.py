from abc import ABC, abstractmethod

import mlflow
from mlflow import log_param, log_metric, set_tracking_uri
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types.schema import TensorSpec, TensorType
from mlflow.models import ModelSignature, ModelInputExample


from src.config import MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



class ExperimentBuilder(ABC):

    def set_tags(self, tags: dict):
        """Set tags for the experiment."""
        self.tags = tags
        mlflow.set_tags(tags)

    @abstractmethod
    def run(self):
        pass

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

    def log_model_input_tensor(self, input_tensor: TensorType):
        """Log a model input tensor to the experiment."""
        mlflow.log_model_input_tensor(input_tensor)

    def log_model_output_tensor(self, output_tensor: TensorType):
        """Log a model output tensor to the experiment."""
        mlflow.log_model_output_tensor(output_tensor)


class MLFlowExperimentBuilder(ExperimentBuilder):
    def __init__(self, experiment_name: str, model_name: str, model_version: int = 1):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model_version = model_version

    def run(self):
        try:
            # Set the experiment name and create it if it doesn't exist
            mlflow.set_experiment(self.experiment_name)
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                client.create_experiment(self.experiment_name)

            # Start a new MLflow run
            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                # Log parameters and metrics here
                log_param("model_name", self.model_name)
                log_param("model_version", self.model_version)

                # Log the model signature (example only, adjust as needed)
                input_schema = Schema([ColSpec("double", "feature1"), ColSpec("double", "feature2")])
                output_schema = Schema([ColSpec("double", "prediction")])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                mlflow.log_model_signature(signature)

                # Log the model (example only, adjust as needed)
                mlflow.log_artifact("path/to/your/model.pkl")

        except MlflowException as e:
            print(f"Error logging to MLflow: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


import os
import sys
import json
import time
import socket
import pickle
import platform
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.pyfunc
import tensorflow as tf

from abc import ABC, abstractmethod
from src import config
from src.config import logger

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


class ILogStrategy(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError("Implement in subclass")

class FullPipelineModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(context.artifacts["postprocessor"], "rb") as f:
            self.postprocessor = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor.transform(model_input)
        y_pred = self.model.predict(X_transformed)
        return self.postprocessor.inverse_transform(y_pred.reshape(-1, 1)).flatten()



class KerasExperimentMlFlowLogger(ILogStrategy):
    def __init__(
        self,
        model,
        history,
        validation_len,
        batch_size,
        elapsed_time=None
    ):
        self.model = model
        self.history = history
        self.validation_len = validation_len
        self.batch_size = batch_size
        self.elapsed_time = elapsed_time

    def plot_history(self):
        plt.figure(figsize=(10, 5))
        for key, values in self.history.history.items():
            plt.plot(values, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        return plt

    def log_environment(self):
        mlflow.set_tag("python_version", platform.python_version())
        mlflow.set_tag("tensorflow_version", tf.__version__)
        mlflow.set_tag("hostname", socket.gethostname())
        mlflow.set_tag("platform", platform.platform())
        mlflow.set_tag("processor", platform.processor())

        gpus = tf.config.list_physical_devices("GPU")
        mlflow.set_tag("gpu_available", bool(gpus))
        mlflow.set_tag("gpu_count", len(gpus))
        mlflow.set_tag("gpu_names", ", ".join([gpu.name for gpu in gpus]) if gpus else "None")

    def run(
        self,
        run_name="training_run",
        experiment_name="default_experiment",
        model_name="regression-pipeline",        
        purpose_tag = "regression-pipeline"
    ):
        mlflow.set_experiment(experiment_name)
        logger.info(f"Starting MLflow run '{run_name}'")

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("framework", "keras")
            mlflow.set_tag("developer", os.getenv("USER", "unknown"))
            mlflow.set_tag("purpose", purpose_tag)

            try:
                commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
                mlflow.set_tag("git_commit", commit_hash)
            except Exception:
                mlflow.set_tag("git_commit", "unavailable")

            mlflow.set_tag("pipeline_version", "1.0.0")
            self.log_environment()

            # Keras model
            mlflow.keras.log_model(self.model, "artifacts/model", registered_model_name=model_name)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Model backup
                model_path = os.path.join(tmpdir, "model.keras")
                try:
                    self.model.save(model_path)
                    mlflow.log_artifact(model_path, artifact_path="artifacts/model_backup")
                except Exception as e:
                    logger.warning(f"Failed to save model backup: {e}")

                # History
                if not hasattr(self.history, "history"):
                    raise ValueError("Invalid history object passed to logger.")

                history_path = os.path.join(tmpdir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(self.history.history, f, indent=2)
                mlflow.log_artifact(history_path, artifact_path="history")

                plot = self.plot_history()
                plot_path = os.path.join(tmpdir, "training_plot.png")
                plot.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots")
                plt.close()

                # Transformadores
                pre = config.PROCESSED_DATA_DIR / "preprocessor.pkl"
                post = config.PROCESSED_DATA_DIR / "postprocessor.pkl"

                if not pre.exists() or not post.exists():
                    raise FileNotFoundError("Preprocessor or postprocessor not found.")

                mlflow.log_artifact(pre, artifact_path="artifacts/processors")
                mlflow.log_artifact(post, artifact_path="artifacts/processors")

                prep_dest = os.path.join(tmpdir, "preprocessor.pkl")
                post_dest = os.path.join(tmpdir, "postprocessor.pkl")
                model_dest = os.path.join(tmpdir, "model.pkl")

                with open(pre, "rb") as f_in, open(prep_dest, "wb") as f_out:
                    f_out.write(f_in.read())
                with open(post, "rb") as f_in, open(post_dest, "wb") as f_out:
                    f_out.write(f_in.read())
                with open(model_dest, "wb") as f:
                    pickle.dump(self.model, f)

                # Validação do input_example
                dataset_path = config.PROCESSED_DATA_DIR / "dataset.csv"
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

                input_example = pd.read_csv(dataset_path, index_col=0)[-(self.batch_size+1):]
                try:
                    _ = pickle.load(open(pre, "rb")).transform(input_example)
                except Exception as e:
                    raise ValueError(f"Invalid input_example: {e}")

                mlflow.pyfunc.log_model(
                    artifact_path="pyfunc_model",
                    python_model=FullPipelineModel(),
                    artifacts={
                        "preprocessor": prep_dest,
                        "postprocessor": post_dest,
                        "model": model_dest,
                    },
                    input_example=input_example,
                    registered_model_name=model_name,
                )

            X_path = config.PROCESSED_DATA_DIR / "X_train.npy"
            if not X_path.exists():
                raise FileNotFoundError(f"Training data not found: {X_path}")

            X_train = np.load(X_path)

            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("validation_len", self.validation_len)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", len(self.history.history.get("loss", [])))
            mlflow.log_param("training_time_sec", self.elapsed_time)

            for metric, values in self.history.history.items():
                if values:
                    mlflow.log_metric(f"final_{metric}", values[-1])
                    for epoch, val in enumerate(values):
                        mlflow.log_metric(metric, val, step=epoch)

            source_file = globals().get("__file__")
            if source_file and os.path.isfile(source_file):
                mlflow.log_artifact(source_file, artifact_path="source_code")

            logger.success("MLflow logging completed.")

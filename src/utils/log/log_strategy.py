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
from mlflow.models.signature import infer_signature
import mlflow.keras
import tensorflow as tf

from abc import ABC, abstractmethod
from src import config
from src.config import logger
from src.utils.log.PythonModelPipeline import PythonModelPipeline


class ILogStrategy(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError("Implement in subclass")


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

    def log_Keras(self, model_name):
        """
        Logs the Keras model to MLflow with the specified model name.
        """
        logger.info("Logging Keras model...")

        X_path = config.PROCESSED_DATA_DIR / "X_train.npy"
        y_path = config.PROCESSED_DATA_DIR / "y_train.npy"

        X_val = np.load(X_path)[-self.validation_len:]
        y_val = np.load(y_path)[-self.validation_len:]

        # Cria a assinatura do modelo
        signature = infer_signature(X_val, y_val)

        # Salva o modelo com a assinatura
        mlflow.keras.log_model(
            self.model,
            artifact_path="artifacts/model",
            registered_model_name=model_name,            
            signature=signature
        )
        

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
    
    def plot_val_predictions(self):

        X_path = config.PROCESSED_DATA_DIR / "X_train.npy"
        y_path = config.PROCESSED_DATA_DIR / "y_train.npy"

        X_val = np.load(X_path)[-self.validation_len:]
        y_val = np.load(y_path)[-self.validation_len:]
        y_pred = self.model.predict(X_val)
        

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        post = config.PROCESSED_DATA_DIR / "postprocessor.pkl"
        postprocessor = pickle.load(open(post, "rb"))

        # y_val and y_pred should be DataFrames after inverse_transform
        y_val = postprocessor.inverse_transform(y_val)
        y_pred = postprocessor.inverse_transform(y_pred)

        # If they are still numpy arrays, convert to DataFrame with matching column names
        if not isinstance(y_val, pd.DataFrame):
            y_val = pd.DataFrame(y_val, columns=[f"Output {i+1}" for i in range(y_val.shape[1])])
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_val.columns)

        # Number of output variables
        n_outputs = y_val.shape[1]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(10, 5 * n_outputs), sharex=True)

        # Ensure axs is iterable even if there is only one plot
        if n_outputs == 1:
            axs = [axs]

        # Plot each output with proper axis labeling
        for i, col in enumerate(y_val.columns):
            axs[i].plot(y_val[col], label="True", color="blue")
            axs[i].plot(y_pred[col], label="Predicted", color="orange")
            axs[i].set_ylabel(col)
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel("Time")
        fig.suptitle("Model Predictions vs True Values on Validation Set")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return plt
    
    def plot_test_predictions(self):

        X_path = config.PROCESSED_DATA_DIR / "X_test.npy"
        y_path = config.PROCESSED_DATA_DIR / "y_test.npy"

        X_test = np.load(X_path)
        y_test = np.load(y_path)
        y_pred = self.model.predict(X_test)
        

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        post = config.PROCESSED_DATA_DIR / "postprocessor.pkl"
        postprocessor = pickle.load(open(post, "rb"))

        # y_val and y_pred should be DataFrames after inverse_transform
        y_test = postprocessor.inverse_transform(y_test)
        y_pred = postprocessor.inverse_transform(y_pred)

        # If they are still numpy arrays, convert to DataFrame with matching column names
        if not isinstance(y_test, pd.DataFrame):
            y_test = pd.DataFrame(y_test, columns=[f"Output {i+1}" for i in range(y_test.shape[1])])
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

        # Number of output variables
        n_outputs = y_test.shape[1]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(10, 5 * n_outputs), sharex=True)

        # Ensure axs is iterable even if there is only one plot
        if n_outputs == 1:
            axs = [axs]

        # Plot each output with proper axis labeling
        for i, col in enumerate(y_test.columns):
            axs[i].plot(y_test[col], label="True", color="blue")
            axs[i].plot(y_pred[col], label="Predicted", color="orange")
            axs[i].set_ylabel(col)
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel("Time")
        fig.suptitle("Model Predictions vs True Values on Test Set")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

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

    def log_metrics(self):
        """
        Logs the training metrics to MLflow.
        """
        logger.info("Logging training metrics...")

        for metric, values in self.history.history.items():
            if values:
                mlflow.log_metric(f"final_{metric}", values[-1])
                for epoch, val in enumerate(values):
                    mlflow.log_metric(metric, val, step=epoch)


        X_test = np.load(config.PROCESSED_DATA_DIR / "X_test.npy")
        y_test = np.load(config.PROCESSED_DATA_DIR / "y_test.npy")

        test_results = self.model.evaluate(X_test, y_test, batch_size=self.batch_size, verbose=0)
       
        metric_names = []
        for m in self.model.metrics:
            if hasattr(m, 'metrics'):
                # if the metric is a container of multiple metrics
                metric_names.extend([subm.name for subm in m.metrics])
            else:
                metric_names.append(m.name)


        for name, value in zip(metric_names, test_results):
            mlflow.log_metric(f"test_{name}", value)

    def log_param(self):
        """
        Logs the training parameters to MLflow.
        """
        logger.info("Logging training parameters...")

        X_path = config.PROCESSED_DATA_DIR / "X_train.npy"

        if not X_path.exists():
            raise FileNotFoundError(f"Training data not found: {X_path}")

        X_train = np.load(X_path)
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("validation_len", self.validation_len)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("epochs", len(self.history.history.get("loss", [])))
        mlflow.log_param("training_time_sec", self.elapsed_time)

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

                plot = self.plot_val_predictions()
                plot_path = os.path.join(tmpdir, "model_output_val_plot.png")
                plot.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots")
                plt.close()

                plot = self.plot_test_predictions()
                plot_path = os.path.join(tmpdir, "model_output_test_plot.png")
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

                input_example = pd.read_csv(dataset_path, index_col=0)

                # Carrega o preprocessor
                with open(pre, "rb") as f:
                    preprocessor = pickle.load(f)

                # Verifica se ele possui o atributo `_generator._sequence_length`
                try:
                    sequence_length = preprocessor._generator._sequence_length
                except AttributeError:
                    raise ValueError("The preprocessor does not expose _generator._sequence_length")

                # Ajusta o input_example com base na sequência
                n_required = sequence_length + 1
                if len(input_example) < n_required:
                    raise ValueError(f"Input example must have at least {n_required} rows.")

                input_example = input_example[-n_required:]

                try:
                    _ = preprocessor.transform(input_example)
                except Exception as e:
                    raise ValueError(f"input_example is not valid for the preprocessor: {e}")

                # self.log_Keras(model_name)

                mlflow.pyfunc.log_model(
                    artifact_path="pyfunc_model",
                    python_model=PythonModelPipeline(),
                    artifacts={
                        "preprocessor": prep_dest,
                        "postprocessor": post_dest,
                        "model": model_dest,
                    },
                    input_example=input_example,
                    registered_model_name=model_name,
                )

            self.log_param()

            self.log_metrics()

            source_file = globals().get("__file__")
            if source_file and os.path.isfile(source_file):
                mlflow.log_artifact(source_file, artifact_path="source_code")

            logger.success("MLflow logging completed.")

import mlflow
import tempfile
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import socket
import platform
import tensorflow as tf
import sys

from src import config
from src.config import logger

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

from abc import ABC, abstractmethod

class LogStrategy(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError("Implement in subclass")

class MLflowLogger(LogStrategy):
    def __init__(
        self,
        model,
        history,
        validation_len,
        batch_size,
        X_train=None,
        y_train=None,
        elapsed_time=None,
    ):
        self.model = model
        self.history = history
        self.validation_len = validation_len
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.elapsed_time = elapsed_time

    def plot_history(self):
        plt.figure(figsize=(10, 5))
        for key in self.history.history:
            plt.plot(self.history.history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        return plt

    def log_environment(self):
        logger.info("Logging environment details...")

        try:
            mlflow.set_tag("python_version", platform.python_version())
            mlflow.set_tag("tensorflow_version", tf.__version__)
            mlflow.set_tag("hostname", socket.gethostname())
            mlflow.set_tag("platform", platform.platform())
            mlflow.set_tag("processor", platform.processor())

            if tf.config.list_physical_devices("GPU"):
                gpus = tf.config.list_physical_devices("GPU")
                mlflow.set_tag("gpu_available", True)
                mlflow.set_tag("gpu_count", len(gpus))
                mlflow.set_tag("gpu_names", ", ".join([gpu.name for gpu in gpus]))
            else:
                mlflow.set_tag("gpu_available", False)

        except Exception as e:
            logger.warning(f"Could not log environment info: {e}")

    def run(
        self,
        run_name="training_run",
        experiment_name="default_experiment",
        model_name="2025-amp-rnn"
    ):
        mlflow.set_experiment(experiment_name)
        logger.info(f"Starting MLflow run '{run_name}' in experiment '{experiment_name}'")

        with mlflow.start_run(run_name=run_name):
            # Tags de contexto e Git
            mlflow.set_tag("framework", "keras")
            mlflow.set_tag("developer", os.getenv("USER", "unknown"))
            mlflow.set_tag("purpose", "lstm-regression")
            mlflow.set_tag("run_type", "baseline")

            try:
                commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
                mlflow.set_tag("git_commit", commit_hash)
            except Exception:
                mlflow.set_tag("git_commit", "unavailable")

            # Ambiente
            self.log_environment()

            # Modelo Keras
            logger.info("Logging Keras model...")
            mlflow.keras.log_model(
                self.model,
                artifact_path="keras_model",
                registered_model_name=model_name
            )

            # Backup .h5
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model_backup.h5")
                self.model.save(model_path)
                mlflow.log_artifact(model_path, artifact_path="model_backup")

            meta = {
                "input_shape": getattr(self.model, "input_shape", None),
                "output_shape": getattr(self.model, "output_shape", None),
                "train_size": len(self.X_train) if self.X_train is not None else 0,
                "validation_len": self.validation_len,
                "batch_size": self.batch_size,
                "epochs_ran": len(self.history.history.get("loss", [])),
                "training_time_sec": self.elapsed_time
            }

            for k, v in meta.items():
                if v is not None:
                    mlflow.log_param(k, v)

            # Métricas finais + agregadas
            logger.info("Logging final and aggregate metrics...")
            for metric_name, values in self.history.history.items():
                if not values:
                    continue
                mlflow.log_metric(f"final_{metric_name}", values[-1])

                for epoch, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=epoch)

            # Artefatos: histórico e gráfico
            logger.info("Logging artifacts...")
            with tempfile.TemporaryDirectory() as tmpdir:
                history_path = os.path.join(tmpdir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(self.history.history, f, indent=2)
                mlflow.log_artifact(history_path, artifact_path="history")

                plot = self.plot_history()
                plot_path = os.path.join(tmpdir, "training_plot.png")
                plot.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots")
                plt.close()

            # Código fonte (opcional)
            try:
                source_path = __file__
                if os.path.isfile(source_path):
                    mlflow.log_artifact(source_path, artifact_path="source_code")
            except Exception as e:
                logger.warning(f"Could not log source code: {e}")

            logger.success("MLflow logging complete.")

import mlflow
import tempfile
import json
import os
import time
import matplotlib.pyplot as plt
from src import config


mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

class MLflowLogger:
    def __init__(
        self,
        model,
        history,
        builder_strategy,
        compile_strategy,
        train_strategy,
        batch_size,
        input_shape,
        output_shape,
        model_version="v1.0.0"
    ):
        self.model = model
        self.history = history
        self.builder_strategy = builder_strategy
        self.compile_strategy = compile_strategy
        self.train_strategy = train_strategy
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_version = model_version

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

    def log_run(self, run_name="training_run"):
        start_time = time.time()

        # Defina o nome do experimento aqui
        mlflow.set_experiment("lstm-ibov-experiment")
        with mlflow.start_run(run_name=run_name):

            # Log modelo Keras
            mlflow.keras.log_model(
                self.model,
                artifact_path="keras_model",
                registered_model_name=self.builder_strategy,
            )

            # Log parâmetros
            mlflow.log_param("builder_strategy", self.builder_strategy)
            mlflow.log_param("compile_strategy", self.compile_strategy)
            mlflow.log_param("train_strategy", self.train_strategy)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("input_shape", str(self.input_shape))
            mlflow.log_param("output_shape", str(self.output_shape))
            mlflow.log_param("epochs_ran", len(self.history.history.get("loss", [])))

            # Versão e controle de fonte
            mlflow.log_param("model_version", self.model_version)

            # Métricas finais
            final_metrics = {
                f"final_{k}": v[-1] for k, v in self.history.history.items()
            }
            for metric, value in final_metrics.items():
                mlflow.log_metric(metric, value)

            # Curva completa por época
            for metric_name, values in self.history.history.items():
                for epoch, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=epoch)

            # Artefatos (history JSON + gráfico)
            with tempfile.TemporaryDirectory() as tmpdir:
                # History JSON
                history_path = os.path.join(tmpdir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(self.history.history, f, indent=2)
                mlflow.log_artifact(history_path, artifact_path="history")

                # Gráfico
                plot = self.plot_history()
                plot_path = os.path.join(tmpdir, "training_plot.png")
                plot.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots")
                plt.close()

            # Tempo de execução
            elapsed = time.time() - start_time
            mlflow.log_metric("training_time_sec", elapsed)

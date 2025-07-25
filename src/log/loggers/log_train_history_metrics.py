from src.log.interfaces import ILogStrategy
from src import config

from mlflow import log_metric

class LogTrainHistoryMetrics(ILogStrategy):
    def __init__(self, history):
        """
        Initializes the logger with the training history and elapsed time.
        """
        self.history = history

    def run(self, **kwargs):
        """
        Logs the training history metrics to MLflow.
        """
        config.logger.info("Logging training history metrics...")

        if not hasattr(self.history, "history"):
            raise ValueError("Invalid history object passed to logger.")

        for metric, values in self.history.history.items():
            if values:
                log_metric(f"final_{metric}", values[-1])
                for epoch, val in enumerate(values):
                    log_metric(metric, val, step=epoch)
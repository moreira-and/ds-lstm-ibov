from src import config
from log.interface import ILogStrategy

import numpy as np
from mlflow import log_metric

class LogTestMetrics(ILogStrategy):
    def __init__(self, model, batch_size, X_test=None, y_test=None):
        """
        Initializes the logger with the model and batch size.
        """
        self.model = model
        self.batch_size = batch_size
        self.X_test = X_test or np.load(config.X_PROCESSED_DATA_TEST_FILE)
        self.y_test = y_test or np.load(config.Y_PROCESSED_DATA_TEST_FILE)
    
    def log(self, **kwargs):
        """ 
        Logs the test metrics to MLflow.
        """

        test_results = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=0)
       
        metric_names = []
        for m in self.model.metrics:
            if hasattr(m, 'metrics'):
                # if the metric is a container of multiple metrics
                metric_names.extend([subm.name for subm in m.metrics])
            else:
                metric_names.append(m.name)

        for name, value in zip(metric_names, test_results):
            log_metric(f"test_{name}", value)
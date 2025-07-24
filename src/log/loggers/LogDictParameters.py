from src.log.interfaces import ILogStrategy
from src import config

from mlflow import log_param

class LogDictParameters(ILogStrategy):
    def __init__(self,*, param_dict: dict):
        """
        Initializes the logger with a dictionary of parameters.
        {"param": "value", ...}
        """
        self.param_dict = param_dict

    def run(self, **kwargs):
        """
        Logs the training parameters to MLflow.
        """
        config.logger.info("Logging training parameters...")
        
        for param, value in self.dict_params.items():
            if value is not None:
                log_param(param, value)
                config.logger.info(f"Logged parameter: {param} = {value}")
            else:
                config.logger.warning(f"Parameter {param} is None, skipping logging.")
from src.log.interfaces import ILogStrategy
from src import config

from mlflow.keras import log_model
from mlflow.models import infer_signature

import numpy as np

class LogKerasModel(ILogStrategy):
        
    def __init__(self, model, model_name, validation_len):
        self.model = model
        self.model_name = model_name
        self.validation_len = validation_len
        
    def run(self, **kwargs):
        """
        Logs the Keras model to MLflow with the specified model name.
        """
        config.logger.info("Logging Keras model...")

        X = np.load(config.X_PROCESSED_DATA_TEST_FILE)[-self.validation_len:]
        y = np.load(config.Y_PROCESSED_DATA_TEST_FILE)[-self.validation_len:]

        # Cria a assinatura do modelo
        signature = infer_signature(X, y)

        # Salva o modelo com a assinatura
        log_model(
            self.model,            
            registered_model_name= self.model_name,
            signature=signature, 
            artifact_path="artifacts/model"
        )
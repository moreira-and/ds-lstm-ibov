from src.config import logger
from src.train.interfaces import IModelBuilder
from tensorflow.keras.models import load_model

class LoadKerasModel(IModelBuilder):
    def __init__(self, model_path):
        self.model_path = model_path

    def build_model(self):
        try:
            return load_model(self.model_path)
        except Exception as e:
            logger.error(f'Error loading model from {self.model_path}: {e}')
            raise
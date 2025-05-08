from abc import ABC, abstractmethod
from typing import Any
from src.config import logger


class PostprocessorStrategy(ABC):
    @abstractmethod
    def inverse_transform(self, X: Any, y: Any = None) -> Any:
        pass


class DefaultPostprocessor(PostprocessorStrategy):
    def __init__(self, preprocessor):
        if hasattr(preprocessor.column_transformer, 'inverse_transform'):
            self._preprocessor = preprocessor.column_transformer
        else:
            self._preprocessor = None
            logger.warning("Preprocessor does not implement inverse_transform.")


    def inverse_transform(self, X: Any, y: Any = None) -> Any:
        logger.info(f"[Postprocessing] Inversing transformation on shape {X.shape}")
        return self._preprocessor.inverse_transform(X) if self._preprocessor else X

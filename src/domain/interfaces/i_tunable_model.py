from abc import ABC, abstractmethod
from typing import Any, Dict

class ITunableModel(ABC):
    @abstractmethod
    def tune(self, data: Any, labels: Any, param_grid: Dict[str, list]) -> Dict[str, float]:
        """
        Performs hyperparameter tuning.

        Args:
            data: Preprocessed training data.
            labels: Corresponding labels.
            param_grid: Dictionary of hyperparameters to test.

        Returns:
            Dictionary with best parameters and corresponding performance metrics.
        """
        pass

from src.domain.interfaces.validators.i_model_validator import ModelValidator
from src.domain.entities.metadata.model_metadata import ModelMetadata

class AccuracyThresholdValidator(ModelValidator):
    def __init__(self, metric: str = "accuracy",threshold: float = 0.85):
        self.metric = metric
        self.threshold = threshold

    def is_valid(self, metadata: ModelMetadata) -> bool:
        return metadata.metrics.get(self.metric, 0.0) >= self.threshold
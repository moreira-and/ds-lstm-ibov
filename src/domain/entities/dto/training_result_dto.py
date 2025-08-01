from typing import Dict

class TrainingResultDto:
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics

from typing import Dict

class FittingResultDto:
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics

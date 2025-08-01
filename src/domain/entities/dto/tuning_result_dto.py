from dataclasses import dataclass
from typing import Dict,Any

@dataclass(frozen=True)
class TuningResultDto:
    best_params: Dict[str, Any]
    best_score: float
    all_results: Dict[str, float]  # e.g., mapping param sets (as str) to scores

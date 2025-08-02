from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class PredictedData:
    data: Any = field(default=None)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None")

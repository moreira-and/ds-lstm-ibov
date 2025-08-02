from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class ModelMetadata:
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            raise TypeError(f"'metadata' must be a dict, got {type(self.metadata).__name__}")
        
        for key in self.metadata:
            if not isinstance(key, str):
                raise TypeError(f"All keys in 'metadata' must be str, found {type(key).__name__}")

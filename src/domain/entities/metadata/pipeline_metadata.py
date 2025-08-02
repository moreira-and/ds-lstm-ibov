from domain.entities.metadata.adapter_metadata import AdapterMetadata
from domain.entities.metadata.model_metadata import ModelMetadata
from dataclasses import dataclass

@dataclass
class PipelineMetadata:
    adapter_metadata: AdapterMetadata
    model_metadata: ModelMetadata

    def __post_init__(self):
        if not isinstance(self.adapter_metadata, AdapterMetadata):
            raise TypeError(f"adapter_metadata must be AdapterMetadata, got {type(self.adapter_metadata)}")
        if not isinstance(self.model_metadata, ModelMetadata):
            raise TypeError(f"model_metadata must be ModelMetadata, got {type(self.model_metadata)}")

from domain.entities.metadata.cleaner_metadata import CleanerMetadata
from domain.entities.metadata.selector_metadata import SelectorMetadata
from domain.entities.metadata.adapter_metadata import AdapterMetadata
from domain.entities.metadata.model_metadata import ModelMetadata
from dataclasses import dataclass

@dataclass
class FullCycleMetadata:

    cleaner_metadata: CleanerMetadata
    selector_metadata: SelectorMetadata
    adapter_metadata: AdapterMetadata
    model_metadata: ModelMetadata

    def __post_init__(self):
        if not isinstance(self.cleaner_metadata, CleanerMetadata):
            raise TypeError(f"cleaner_metadata must be CleanerMetadata, got {type(self.cleaner_metadata)}")
        if not isinstance(self.selector_metadata, SelectorMetadata):
            raise TypeError(f"selector_metadata must be SelectorMetadata, got {type(self.selector_metadata)}")
        if not isinstance(self.adapter_metadata, AdapterMetadata):
            raise TypeError(f"adapter_metadata must be AdapterMetadata, got {type(self.adapter_metadata)}")
        if not isinstance(self.model_metadata, ModelMetadata):
            raise TypeError(f"model_metadata must be ModelMetadata, got {type(self.model_metadata)}")

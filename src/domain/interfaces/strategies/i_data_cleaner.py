from src.domain.entities.data.raw_data import RawData
from src.domain.entities.data.cleaned_data import CleanedData
from src.domain.entities.metadata.cleaner_metadata import CleanerMetadata
from abc import ABC, abstractmethod

class IDataCleaner(ABC):
    
    """
    Interface for implementing data cleaning strategies.

    This abstraction standardizes the orchestration at the application layer and 
    decouples the implementation details from external libraries or frameworks 
    in the infrastructure layer.

    Methods:
        fit(data: RawData) -> CleanerMetadata:
            Analyzes the raw data and generates metadata that describes the cleaning process.
            This may include rules, thresholds, or transformation logic inferred from the data.

        clean(data: RawData) -> CleanedData:
            Applies the cleaning logic—either predefined or learned in 'fit'—to the raw data
            and returns a cleaned version ready for downstream processing.
    """

    @abstractmethod
    def fit(self, data: RawData) -> CleanerMetadata:
        pass

    @abstractmethod
    def clean(self, data: RawData) -> CleanedData:
        pass
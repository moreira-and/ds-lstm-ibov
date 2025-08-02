from src.domain.entities.data.cleaned_data import CleanedData
from src.domain.entities.data.selected_data import SelectedData
from src.domain.entities.metadata.selector_metadata import SelectorMetadata
from abc import ABC, abstractmethod

class IDataSelector(ABC):

    @abstractmethod
    def fit(self, data: CleanedData) -> SelectorMetadata:
        pass

    @abstractmethod
    def select(self, data: CleanedData) -> SelectedData:
        pass
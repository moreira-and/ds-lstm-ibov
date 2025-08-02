from src.domain.entities.data.raw_data import RawData
from src.domain.entities.data.selected_data import SelectedData
from src.domain.entities.metadata.selector_metadata import SelectorMetadata
from src.domain.interfaces.strategies.i_data_selector import IDataSelector


class MockDataSelector(IDataSelector):
    """
    A jumper/mock implementation of IDataSelector.
    Useful for testing, composition, or bypassing selection logic in pipelines.
    """

    def fit(self, data: RawData) -> SelectorMetadata:
        # Mock metadata output
        metadata = {"status": "pass", "note": "MockDataSelector used"}
        return SelectorMetadata(metadata=metadata)

    def select(self, data: SelectedData) -> SelectedData:
        # Pass-through — no transformation applied
        return SelectedData(data.data)

from src.domain.entities.data.raw_data import RawData
from src.domain.entities.data.cleaned_data import CleanedData
from src.domain.entities.metadata.cleaner_metadata import CleanerMetadata
from src.domain.interfaces.strategies.i_data_cleaner import IDataCleaner


class MockDataCleaner(IDataCleaner):
    """
    A jumper/mock implementation of IDataCleaner.
    Useful for testing, composition, or bypassing selection logic in pipelines.
    """

    def fit(self, data: RawData) -> CleanerMetadata:
        # Mock metadata output
        metadata = {"status": "pass", "note": "MockDataCleaner used"}
        return CleanerMetadata(metadata=metadata)

    def clean(self, data: RawData) -> CleanedData:
        CleanedData(data.data)
from src.domain.entities.data.raw_data import RawData
from src.domain.entities.data.cleaned_data import CleanedData

from src.domain.interfaces.strategies.i_data_cleaner import IDataCleaner


class CleanUseCase:

    def __init__(
        self,
        cleaner: IDataCleaner
    ):
        self.cleaner = cleaner

    def execute(self, data: RawData) -> CleanedData:
        
        metadata = self.cleaner.fit(self, data)
        return self.cleaner.clean(data) # -> CleanedData:
        





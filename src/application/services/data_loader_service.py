from src.domain.entities.data.raw_data import RawData

class DataLoaderService:
    def __init__(self, raw_data_loader: RawDataLoader):
        self.raw_data_loader = raw_data_loader

    def load_raw_data(self, source: str) -> RawData:
        # Aqui pode ter logs, validações, tratamento
        raw_data = self.raw_data_loader.load(source)
        return raw_data

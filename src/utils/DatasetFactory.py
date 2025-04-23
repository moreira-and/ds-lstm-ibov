from dataset_loader import DatasetLoader
from csv_loader import CSVLoader
from sql_loader import SQLLoader
from api_loader import APILoader

class DatasetFactory:
    @staticmethod
    def get_loader(source_type: str, **kwargs) -> DatasetLoader:
        source_type = source_type.lower()
        
        if source_type == "csv":
            return CSVLoader(filepath=kwargs["filepath"])
        elif source_type == "sql":
            return SQLLoader(connection_string=kwargs["connection_string"], query=kwargs["query"])
        elif source_type == "api":
            return APILoader(endpoint=kwargs["endpoint"])
        else:
            raise ValueError(f"Tipo de fonte desconhecida: {source_type}")

from abc import ABC, abstractmethod

import pandas as pd


class DatasetLoader(ABC):
    @abstractmethod
    def load(self):
        """
        Método obrigatório que todas as classes que herdam devem implementar.
        Deve retornar um dataset (ex: DataFrame, lista de dicionários, etc.).
        """
        pass


class CSVLoader(DatasetLoader):
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        return pd.read_csv(self.filepath)
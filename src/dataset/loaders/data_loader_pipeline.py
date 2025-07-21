from config import logger
from dataset.interfaces import IDatasetLoader
from typing import List, Dict
import pandas as pd

class DataLoaderPipeline(IDatasetLoader):
    def __init__(self, strategies: List[IDatasetLoader]):
        self.strategies = strategies

    def load(self) -> Dict[str, pd.DataFrame]:
            
        dataset = {}

        for strategy in self.strategies:
            name = strategy.__class__.__name__.replace("LoadingStrategy","")
            try:
                dataset[name] = strategy.load()
            except Exception as e:
                logger.error(f"Error loading data with {name}: {e}")
                pass
        return dataset
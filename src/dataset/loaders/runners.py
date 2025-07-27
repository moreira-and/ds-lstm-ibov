"""
DataLoaderPipeline module

Defines a composite data loader that orchestrates multiple dataset loading strategies,
aggregating their results into a single dictionary keyed by loader class names.

This allows flexible and modular loading of datasets from various sources
in a unified manner, with error handling and logging per loader.

Classes
-------
DataLoaderPipeline
    Implements IDatasetLoaderStrategy by combining multiple loaders,
    returning a dictionary of named datasets.
"""

from config import logger
from interfaces import IDatasetLoaderStrategy
from typing import List, Dict
import pandas as pd


class DataLoaderPipeline(IDatasetLoaderStrategy):
    """
    Composite data loader that aggregates data from multiple loading strategies.

    Parameters
    ----------
    loaders : List[IDatasetLoaderStrategy]
        List of dataset loader strategies to be executed.

    Methods
    -------
    load() -> Dict[str, pd.DataFrame]
        Executes all loaders and returns a dictionary mapping loader names to their loaded datasets.
        Errors during individual loads are logged but do not stop the pipeline.
    """

    def __init__(self, loaders: List[IDatasetLoaderStrategy]):
        self.loaders = loaders

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Loads datasets from all configured loading strategies.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary mapping each loader's simplified class name to its loaded dataset.

        Notes
        -----
        If any loader raises an exception, it is caught and logged.
        The pipeline continues loading remaining datasets.
        """
        dataset = {}

        for strategy in self.loaders:
            name = strategy.__class__.__name__.replace("Loading", "")
            try:
                dataset[name] = strategy.load()
            except Exception as e:
                logger.error(f"Error loading data with {name}: {e}")
                pass

        return dataset

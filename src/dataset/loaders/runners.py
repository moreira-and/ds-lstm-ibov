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

from typing import List, Dict, Optional
from src.config import logger
from .interfaces import IDatasetLoaderStrategy
from src.dataset.uploaders import AzureBlobStorageLoader
import pandas as pd


class DataLoaderPipeline(IDatasetLoaderStrategy):
    """
    Composite data loader that aggregates data from multiple loading strategies.

    Parameters
    ----------
    loaders : List[IDatasetLoaderStrategy]
        List of dataset loader strategies to be executed.
    blob_storage : Optional[AzureBlobStorageLoader]
        Azure Blob Storage loader for saving data in different layers.

    Methods
    -------
    load() -> Dict[str, pd.DataFrame]
        Executes all loaders and returns a dictionary mapping loader names to their loaded datasets.
        Errors during individual loads are logged but do not stop the pipeline.
    """

    def __init__(self, loaders: List[IDatasetLoaderStrategy], blob_storage: Optional[AzureBlobStorageLoader] = None):
        self.loaders = loaders
        self.blob_storage = blob_storage

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Loads datasets from all configured loading strategies and optionally saves to blob storage.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary mapping each loader's simplified class name to its loaded dataset.

        Notes
        -----
        If any loader raises an exception, it is caught and logged.
        The pipeline continues loading remaining datasets.
        If blob_storage is configured, data will be saved to the raw layer.
        """
        dataset = {}

        for strategy in self.loaders:
            name = strategy.__class__.__name__.replace("Loader", "")
            try:
                data = strategy.load()
                dataset[name] = data
                
                # Save to blob storage if configured
                if self.blob_storage and data:
                    for key, df in data.items():
                        try:
                            self.blob_storage.save_to_layer(df, 'raw', f"{name.lower()}_{key.lower()}", 'parquet')
                        except Exception as e:
                            logger.error(f"Error saving {key} to blob storage: {e}")
                            
            except Exception as e:
                logger.error(f"Error loading data with {name}: {e}")
                pass

        # Process and save to silver layer if configured
        if self.blob_storage and dataset:
            try:
                # Combine all datasets
                combined_df = pd.concat(
                    [df for data_dict in dataset.values() for df in data_dict.values()],
                    axis=1
                )
                
                # Save to silver layer
                self.blob_storage.save_to_layer(combined_df, 'silver', 'combined_dataset', 'parquet')
                
            except Exception as e:
                logger.error(f"Error processing and saving to silver layer: {e}")

        return dataset

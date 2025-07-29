"""
IBGE data loader module.

This module provides functionality to load IPCA data from IBGE API.
"""

from typing import Dict
import pandas as pd
import requests
from ..loaders.interfaces import IDatasetLoaderStrategy
from ...config import logger
from ...dataset.uploaders.azure_blob import AzureBlobStorageLoader

class IBGELoader(IDatasetLoaderStrategy):
    """
    Loader for IBGE (Instituto Brasileiro de Geografia e Estatística) data.
    
    Downloads IPCA (Índice Nacional de Preços ao Consumidor Amplo) data from IBGE API
    and saves it to blob storage.
    """
    
    URL = 'https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/all/d/v2266%202'
    
    def __init__(self, blob_storage: AzureBlobStorageLoader):
        self.blob_storage = blob_storage
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Downloads and loads IBGE IPCA data.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing the loaded IPCA data
        """
        try:
            logger.info("[IBGE] Downloading IPCA data...")
            response = requests.get(self.URL, timeout=60)
            
            if response.ok:
                # Load JSON data into DataFrame
                df = pd.DataFrame(response.json())
                
                # Save to raw layer
                self.blob_storage.save_to_layer(df, 'raw', 'ibge_ipca', 'parquet')
                
                return {"ipca": df}
            else:
                logger.error(f"[IBGE] Failed to download (HTTP status {response.status_code})")
                return {}
                
        except Exception as e:
            logger.error(f"[IBGE] Error: {str(e)}")
            return {}

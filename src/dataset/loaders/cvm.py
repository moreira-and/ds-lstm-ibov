"""
CVM data loader module.

This module provides functionality to load data from CVM (Comissão de Valores Mobiliários).
"""

from typing import Dict
import pandas as pd
import requests
from ..loaders.interfaces import IDatasetLoaderStrategy
from ...config import logger
from ...dataset.uploaders.azure_blob import AzureBlobStorageLoader

class CVMLoader(IDatasetLoaderStrategy):
    """
    Loader for CVM (Comissão de Valores Mobiliários) data.
    
    Downloads investment fund registration data from CVM and saves it to blob storage.
    """
    
    URL = 'https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv'
    
    def __init__(self, blob_storage: AzureBlobStorageLoader):
        self.blob_storage = blob_storage
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Downloads and loads CVM data.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing the loaded CVM data
        """
        try:
            logger.info("[CVM] Downloading investment funds database...")
            response = requests.get(self.URL, timeout=60)
            
            if response.ok:
                # Load data into DataFrame
                df = pd.read_csv(response.content)
                
                # Save to raw layer
                self.blob_storage.save_to_layer(df, 'raw', 'cvm_cad_fi', 'csv')
                
                return {"cvm_funds": df}
            else:
                logger.error(f"[CVM] Failed to download (HTTP status {response.status_code})")
                return {}
                
        except Exception as e:
            logger.error(f"[CVM] Error: {str(e)}")
            return {}

"""
Azure Blob Storage uploader module for handling data storage in different layers.

This module provides functionality to load and save data to Azure Blob Storage
in different layers (raw, bronze, silver) following the medallion architecture pattern.
"""

from typing import Optional, Dict
import pandas as pd
from azure.storage.blob import BlobServiceClient
from src.dataset.loaders.interfaces import IDatasetLoaderStrategy
from src.config import logger

class AzureBlobStorageLoader(IDatasetLoaderStrategy):
    """
    Handles data loading and saving operations with Azure Blob Storage.
    
    Parameters
    ----------
    account_url : str
        Azure Storage Account URL
    container_name : str
        Name of the blob container
    credential : str
        SAS token for authentication
    """
    
    def __init__(self, account_url: str, container_name: str, credential: str):
        self.account_url = account_url
        self.container_name = container_name
        self.credential = credential
        logger.info(f"Initializing Azure Blob Storage client for container: {container_name}")
        
        try:
            # Criar o cliente do blob service usando SAS token
            # Garantir que o token SAS comece com "?"
            sas_token = self.credential if self.credential.startswith("?") else f"?{self.credential}"
            self.blob_service_client = BlobServiceClient(
                account_url=self.account_url,
                credential=sas_token
            )
            logger.info("Successfully created blob service client")
            
            logger.info(f"Attempting to connect to container: {self.container_name}")
            logger.debug(f"Using account URL: {self.account_url}")
            
            # Verificar se o container existe
            self.container_client = self.blob_service_client.get_container_client(container_name)
            logger.info("Container client created successfully")
            
            # Tentar listar blobs para verificar acesso
            try:
                blobs = list(self.container_client.list_blobs())
                logger.info(f"Successfully connected to container. Found {len(blobs)} blobs.")
            except Exception as e:
                logger.error(f"Error listing blobs: {str(e)}")
                logger.debug("Container permissions error - checking if container exists")
                # Tentar pelo menos verificar se o container existe
                properties = self.container_client.get_container_properties()
                logger.info(f"Container exists but might have limited permissions. Last modified: {properties.last_modified}")
            
        except Exception as e:
            logger.error(f"Error initializing Azure Blob Storage: {str(e)}")
            raise
    
    def save_to_layer(self, data: pd.DataFrame, layer: str, name: str, file_format: str = 'parquet'):
        """
        Saves data to a specific layer in blob storage.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to be saved
        layer : str
            Storage layer (raw, bronze, silver)
        name : str
            Name of the file
        file_format : str, optional
            Format to save the file (default: 'parquet')
        """
        try:
            blob_path = f"{layer}/{name}.{file_format}"
            logger.info(f"Attempting to save blob to path: {blob_path}")
            
            # Verificar se o container ainda está acessível
            try:
                self.container_client.list_blobs(name_starts_with=layer)
                logger.info("Container is accessible")
            except Exception as e:
                logger.error(f"Container is not accessible: {str(e)}")
                raise
            
            blob_client = self.container_client.get_blob_client(blob_path)
            
            if file_format == 'parquet':
                buffer = data.to_parquet()
            elif file_format == 'csv':
                buffer = data.to_csv(index=False).encode()
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            blob_client.upload_blob(buffer, overwrite=True)
            logger.info(f"Saved {name} to {layer} layer successfully")
            
        except Exception as e:
            logger.error(f"Error saving to blob storage: {e}")
            raise
    
    def load_from_layer(self, layer: str, name: str, file_format: str = 'parquet') -> Optional[pd.DataFrame]:
        """
        Loads data from a specific layer in blob storage.
        
        Parameters
        ----------
        layer : str
            Storage layer to load from (raw, bronze, silver)
        name : str
            Name of the file
        file_format : str, optional
            Format of the file (default: 'parquet')
            
        Returns
        -------
        Optional[pd.DataFrame]
            Loaded data or None if file doesn't exist
        """
        try:
            blob_path = f"{layer}/{name}.{file_format}"
            blob_client = self.container_client.get_blob_client(blob_path)
            
            stream = blob_client.download_blob()
            
            if file_format == 'parquet':
                return pd.read_parquet(stream.readall())
            elif file_format == 'csv':
                return pd.read_csv(stream.readall())
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error loading from blob storage: {e}")
            return None
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Implements the IDatasetLoaderStrategy interface.
        This method should be called by subclasses that implement specific loading logic.
        """
        raise NotImplementedError("This is a base class for blob storage operations. Use specific implementations.")

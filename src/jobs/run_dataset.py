import time

from loguru import logger
import typer
import os

from src.utils import ConfigWrapper
from config.paths import MAIN_RAW_FILE, DATASET_PARAMS_FILE

from dataset.loaders.runners import DataLoaderPipeline
from dataset.loaders import (
    YfinanceLoader, 
    BcbLoader, 
    DataReaderLoader,
    CVMLoader,
    IBGELoader
)
from dataset.uploaders import AzureBlobStorageLoader
from dataset.helpers.calendar import enrich_calendar

import datetime as dt
import pandas as pd

app = typer.Typer()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Starting data loading pipeline...")

    # Carregar configurações
    dataset_param = ConfigWrapper(DATASET_PARAMS_FILE)
    years = dataset_param.get("years")
        
    end_date = dt.datetime.now().date()
    start_date = (end_date - dt.timedelta(days=years*365))
    
    logger.info(f'Requesting information between {start_date} and {end_date}')
    
    # Configurar Azure Blob Storage
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    
    if not connection_string or not container_name:
        raise ValueError("Azure Blob Storage connection string and container name must be set in environment variables")
    
    blob_storage = AzureBlobStorageLoader(connection_string, container_name)
    
    # Configurar e executar pipeline
    loaders = DataLoaderPipeline(
        loaders=[
            YfinanceLoader(start_date, end_date),
            BcbLoader(start_date, end_date),
            DataReaderLoader(start_date, end_date),
            CVMLoader(blob_storage),
            IBGELoader(blob_storage)
        ],
        blob_storage=blob_storage
    )

    # Carregar dados - isso vai automaticamente salvar no blob
    dict_raw = loaders.load()

    # Carregar o dataset combinado da camada silver do blob
    try:
        df_raw = blob_storage.load_from_layer('silver', 'combined_dataset')
        if df_raw is None:
            logger.error("Failed to load combined dataset from silver layer")
            return
            
        # Enriquecer com dados de calendário
        df_raw = enrich_calendar(df_raw)
        
        # Salvar versão final localmente (se necessário)
        if MAIN_RAW_FILE:
            df_raw.to_csv(MAIN_RAW_FILE)
            logger.info(f"Saved enriched dataset to {MAIN_RAW_FILE}")
        
        print("Preview of the final dataset:")
        print(df_raw.tail(3))
        
        # Salvar versão enriquecida de volta no blob
        blob_storage.save_to_layer(df_raw, 'silver', 'enriched_dataset')
        logger.success("Data pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing final dataset: {e}")
        raise

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
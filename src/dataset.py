from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.dataset_loader import MultiLoader
from src.utils.clean_handler import CleanPipeline

import datetime as dt

import pandas as pd



app = typer.Typer()


@app.command()
def main(
    # ----  DEFAULT PATHS --------------------------
    asset: str = '^BVSP',
    asset_focus: str = 'Close',
    years: int = 10
    # ----------------------------------------------
):
    # -----------------------------------------
    logger.info("Starting raw data loading...")
    end_date = dt.datetime.now().date()
    start_date = (end_date - dt.timedelta(days=years*365))
    

    logger.info(f'Requesting information between {start_date} and {end_date}')
    try:
        dict_raw = MultiLoader(start_date,end_date).load()

        df_raw = pd.DataFrame()

        for lib,dict in dict_raw.items():
            for name, df in dict.items():
                output_path = RAW_DATA_DIR / f"{lib}_{name}.csv"
                df.to_csv(output_path, index=True)
                df_raw = pd.concat([df_raw,df],axis=1)
                logger.info(f"Saved {lib}_{name} dataset to {output_path}")

    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
    
    df_raw.to_csv(RAW_DATA_DIR / 'dataset.csv')
    df_raw = pd.read_csv(RAW_DATA_DIR / 'dataset.csv', index_col=0) # ensure coupling

    logger.success("Raw data successfully loaded...")

    # Supondo df como seu DataFrame
    target_cols = [col for col in df_raw.columns if asset in col]
    target_col = [col for col in target_cols if asset_focus in col]
    y = df_raw[target_col]  # This column is kept only for cleaning purposes.
    X = df_raw.drop(columns=target_col) 
    X_clean, y_clean = CleanPipeline().clear(X, y)

    df_clean = pd.concat([X_clean,y_clean],axis=1)

    df_clean.to_csv(PROCESSED_DATA_DIR / 'dataset.csv')
    logger.success("Clean data successfully loaded...")


    # -----------------------------------------


if __name__ == "__main__":
    app()

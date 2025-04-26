from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, CONFIG_DIR
from src.utils.dataset_loader import MultiLoader

import datetime as dt

import pandas as pd

from sklearn.model_selection import train_test_split

app = typer.Typer()


@app.command()
def main(
    # ----  DEFAULT PATHS --------------------------
    # target: List[] = ['']
    # ----------------------------------------------
):
    # -----------------------------------------
    logger.info("Starting raw data loading...")
    years = 10
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

    logger.success("Raw data successfully loaded...")  # se estiver usando `loguru`

    df_cleaned = CleanHandler().clear(df_raw)

    df_train, df_test = ModelHandler().train_test_split(df_cleaned)

    logger.info("Packing raw dataset...")
    # logger.info("Splitting dataset into training and test sets...")

    # X,y = slipt...
    # -----------------------------------------


if __name__ == "__main__":
    app()

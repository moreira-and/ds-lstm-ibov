import time

from loguru import logger
import typer

from src.utils import ConfigWrapper
from config.paths import MAIN_RAW_FILE, RAW_DATA_DIR, DATASET_PARAMS_FILE

from dataset.loaders.runners import DataLoaderPipeline
from dataset.loaders import YfinanceLoader, BcbLoader, DataReaderLoader
from dataset.helpers.calendar import enrich_calendar

import datetime as dt
import pandas as pd

app = typer.Typer()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Starting raw data loading...")

    dataset_param = ConfigWrapper(DATASET_PARAMS_FILE)
    years = dataset_param.get("years")
        
    end_date = dt.datetime.now().date()
    start_date = (end_date - dt.timedelta(days=years*365))
    
    logger.info(f'Requesting information between {start_date} and {end_date}')
    
    df_raw = pd.DataFrame()

    #try:
    loaders = DataLoaderPipeline([
        YfinanceLoader(start_date, end_date),
        BcbLoader(start_date, end_date),
        DataReaderLoader(start_date, end_date)
    ])

    dict_raw = loaders.load()    

    for lib,dict in dict_raw.items():
        for name, df in dict.items():
            output_path = RAW_DATA_DIR / f"{lib}_{name}.csv"
            df.to_csv(output_path, index=True)
            df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in df.columns]
            df_raw = pd.concat([df_raw,df],axis=1)
            logger.info(f"Saved {lib}_{name} dataset to {output_path}")

    #except Exception as e:
    #    logger.error(f"Error loading raw data: {e}")
    
    #df_raw = enrich_calendar(df_raw)

    df_raw = df_raw.ffill().bfill().fillna(0)

    df_raw.to_csv(MAIN_RAW_FILE)
    print(df_raw.tail(3))

    logger.success("Raw data successfully loaded...")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
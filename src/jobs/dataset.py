import time

from loguru import logger
import typer

from src.config.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.dataset.calendar_strategy import update_df, enrich_calendar

from src.utils.dataset.dataset_loading_strategy import DatasetMultiLoader, YfinanceLoadingStrategy, BcbLoadingStrategy, DataReaderLoadingStrategy
from src.utils.dataset.clean_strategy import CleanPipeline, CleanMissingValues, CleanLowVariance, CleanGenericUnivariate

from sklearn.feature_selection import (f_classif, f_regression)

import datetime as dt
import pandas as pd
import numpy as np



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
    start_time = time.time()
    logger.info("Starting raw data loading...")
    
    end_date = dt.datetime.now().date()
    start_date = (end_date - dt.timedelta(days=years*365))
    

    logger.info(f'Requesting information between {start_date} and {end_date}')
    
    try:
        loaders = DatasetMultiLoader([
            YfinanceLoadingStrategy(start_date, end_date),
            BcbLoadingStrategy(start_date, end_date),
            DataReaderLoadingStrategy(start_date, end_date)
        ])

        dict_raw = loaders.load()

        df_raw = pd.DataFrame()

        for lib,dict in dict_raw.items():
            for name, df in dict.items():
                output_path = RAW_DATA_DIR / f"{lib}_{name}.csv"
                df.to_csv(output_path, index=True)
                df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in df.columns]
                df_raw = pd.concat([df_raw,df],axis=1)
                logger.info(f"Saved {lib}_{name} dataset to {output_path}")

    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
    
    df_raw.to_csv(RAW_DATA_DIR / 'dataset.csv')
    
    df_raw = pd.read_csv(RAW_DATA_DIR / 'dataset.csv', index_col=0) # ensure coupling

    print(df_raw.tail(3))

    logger.success("Raw data successfully loaded...")

    # Supondo df como seu DataFrame
    target_cols = [col for col in df_raw.columns if asset in col]
    target_col = [col for col in target_cols if asset_focus in col]
    
    df_raw = df_raw.dropna(subset=target_col).sort_index()
    df_raw = df_raw.ffill().bfill()
    df_raw = df_raw.pct_change(periods=1,fill_method=None)
    df_raw = df_raw.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(df_raw.tail(3))

    '''
    y = df_raw[target_col]  # This column is kept only for cleaning purposes.
    X = df_raw.drop(columns=target_col)
   
    clean_pipeline = CleanPipeline([
            CleanMissingValues(),
            CleanLowVariance(),
            CleanGenericUnivariate(f_classif, "percentile", 99),    
            CleanGenericUnivariate(f_regression, "percentile", 99)
            ])

    X_clean, y_clean = clean_pipeline.clear(X, y)

    df_clean = pd.concat([y_clean,X_clean],axis=1)
    
    dataset_path = PROCESSED_DATA_DIR / 'dataset.csv'
    
    df_old_dataset = pd.read_csv(dataset_path, index_col=0) if dataset_path.exists() else df_clean.copy()

    print(df_old_dataset.tail(7))

    df_updated = update_df(df_old_dataset, df_clean)
    '''

    df_dataset = enrich_calendar(df_raw)
    
    print(df_dataset.tail(3))

    df_dataset.to_csv(PROCESSED_DATA_DIR / 'dataset.csv')
    logger.success("Clean data successfully loaded...")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
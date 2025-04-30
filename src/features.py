import time
from pathlib import Path

import joblib
from tqdm import tqdm
import typer

from src.config import logger, PROCESSED_DATA_DIR

from utils.features.prepare_data_template import LstmPrepareDataTemplate
from utils.features.splitter_strategy import SequentialSplitter
from utils.features.preprocessor_strategy import DefaultPreprocessor
from utils.features.generator_strategy import TimeseriesGenerator

import numpy as np
import pandas as pd


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    train_dir: Path = PROCESSED_DATA_DIR,
    test_dir: Path = PROCESSED_DATA_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()
    logger.info("Generating features from dataset...")

    try:
        prepare_data_template = LstmPrepareDataTemplate(
            dataset = pd.read_csv(input_path, index_col=0).sort_index(),
            splitter=SequentialSplitter(),
            preprocessor=DefaultPreprocessor(),
            generator = TimeseriesGenerator()
            )
        
        prepare_data_template.prepare_data()

        X_train,X_test,y_train, y_test = prepare_data_template.get_data()

        logger.success(f"Saving train features in {train_dir}...")
        np.save(train_dir / 'X_train.npy',X_train)
        np.save(train_dir / 'y_train.npy',y_train)

        logger.success(f"Saving test features in {test_dir}...")
        np.save(train_dir / 'X_test.npy',X_test)
        np.save(test_dir / 'y_test.npy',y_test)
    
        logger.success("Features generation complete.")

        joblib.dump(prepare_data_template.preprocessor, train_dir / 'preprocessor.joblib')
        joblib.dump(prepare_data_template.generator, train_dir / 'generator.joblib')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")

    # -----------------------------------------


if __name__ == "__main__":
    app()

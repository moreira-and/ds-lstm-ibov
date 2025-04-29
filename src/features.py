import time
from pathlib import Path

import joblib
from tqdm import tqdm
import typer

from src.config import logger, PROCESSED_DATA_DIR

from src.utils.prepare_strategy import LstmPrepareTemplate
from src.utils.splitter_strategy import SequentialSplitter
from src.utils.preprocessor_strategy import DefaultPreprocessor
from src.utils.generator_strategy import TimeseriesGeneratorStrategy

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
        prepare_template = LstmPrepareTemplate(
            dataset = pd.read_csv(input_path, index_col=0).sort_index(),
            splitter=SequentialSplitter(),
            preprocessor=DefaultPreprocessor(),
            generator = TimeseriesGeneratorStrategy()
            )
        
        prepare_template.prepare_data()

        X_train,X_test,y_train, y_test = prepare_template.get_data()

        logger.success(f"Saving train features in {train_dir}...")
        np.save(train_dir / 'X_train.npy',X_train)
        np.save(train_dir / 'y_train.npy',y_train)

        logger.success(f"Saving test features in {test_dir}...")
        np.save(train_dir / 'X_test.npy',X_test)
        np.save(test_dir / 'y_test.npy',y_test)
    
        logger.success("Features generation complete.")

        joblib.dump(prepare_template.preprocessor, train_dir / 'preprocessor.joblib')
        joblib.dump(prepare_template.generator, train_dir / 'generator.joblib')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")

    # -----------------------------------------


if __name__ == "__main__":
    app()

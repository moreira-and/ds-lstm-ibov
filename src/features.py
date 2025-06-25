import time
from pathlib import Path

import cloudpickle
import typer
from typing import List

from src.config import logger, PROCESSED_DATA_DIR

from src.utils.features.prepare_data_template import DefaultLstmPrepareDataTemplate
from src.utils.features.splitter_strategy import SequentialSplitter
from src.utils.features.transform_strategy import (DefaultLstmTransformStrategy,BlankTransformStrategy)
from src.utils.features.generator_strategy import DefaultLstmGenerator

import numpy as np
import pandas as pd


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    train_dir: Path = PROCESSED_DATA_DIR,
    test_dir: Path = PROCESSED_DATA_DIR,
    targets: List[str] = ["^BVSP"],
    sequence_length: int = 20
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()
    logger.info("Generating features from dataset...")

    try:
        prepare_data_template = DefaultLstmPrepareDataTemplate(
            dataset = pd.read_csv(input_path, index_col=0).sort_index(),
            targets = targets,
            splitter =SequentialSplitter(train_size_ratio=1),
            transformer = DefaultLstmTransformStrategy(), #BlankTransformStrategy(),
            generator = DefaultLstmGenerator(batch_size=1,sequence_length=sequence_length)
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

        logger.info("Saving transformers...")
        
        preprocessor = prepare_data_template.get_preprocessor()
        with open(train_dir / "preprocessor.pkl", "wb") as f:
            cloudpickle.dump(preprocessor, f)

        postprocessor = prepare_data_template.get_postprocessor()
        with open(train_dir / "postprocessor.pkl", "wb") as f:
            cloudpickle.dump(postprocessor, f)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")

    # -----------------------------------------


if __name__ == "__main__":
    app()

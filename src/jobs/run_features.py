from config import logger
from config.paths import (
    PROCESS_PARAMS_FILE,
    MAIN_RAW_FILE, TEST_RAW_FILE, MAIN_PROCESSED_FILE,
    MAIN_PREPROCESSOR_FILE, MAIN_POSTPROCESSOR_FILE,
    X_PROCESSED_DATA_TRAIN_FILE,X_PROCESSED_DATA_TEST_FILE,
    Y_PROCESSED_DATA_TRAIN_FILE,Y_PROCESSED_DATA_TEST_FILE
    )

from utils.config_wrapper import ConfigWrapper

from features.preprocessors.runners import RnnDataPreparationPipeline
from features.preprocessors.rnn_processors import *
from features.splitters import SequentialSplitter

import cloudpickle

import time
import numpy as np
import pandas as pd


import typer
app = typer.Typer()


@app.command()
def main():

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()

    features_wapper = ConfigWrapper(PROCESS_PARAMS_FILE)

    targets = features_wapper.get("targets")
    train_size_ratio = features_wapper.get("train_size_ratio")
    batch_size = features_wapper.get("batch_size")
    sequence_length = features_wapper.get("sequence_length")

    logger.info("Generating features from dataset...")

    try:

        ## SUBSTITUIR POR LEITURA DO BLOB -> DADO EM MEMÓRIA (ReadFromBlob(), on helpers in dataset module)
        dataset = pd.read_csv(MAIN_RAW_FILE, index_col=0).sort_index()

        prepare_data_template = RnnDataPreparationPipeline(
            dataset = dataset,
            targets = targets,
            splitter =SequentialSplitter(train_size_ratio=train_size_ratio),
            transformer = DefaultRnnTransformStrategy(),
            generator = DefaultRnnGenerator(batch_size=batch_size,sequence_length=sequence_length)
        )
        
        prepare_data_template.prepare_data()

        X_train,X_test,y_train, y_test = prepare_data_template.get_data()

        logger.success(f"Saving train features...")
        np.save(X_PROCESSED_DATA_TRAIN_FILE,X_train)
        np.save(Y_PROCESSED_DATA_TRAIN_FILE,y_train)

        logger.success(f"Saving test features...")
        np.save(X_PROCESSED_DATA_TEST_FILE,X_test)
        np.save(Y_PROCESSED_DATA_TEST_FILE,y_test)
    
        logger.success("Features generation complete.")

        logger.info("Saving transformers...")
        
        preprocessor = prepare_data_template.get_preprocessor()
        with open(MAIN_PREPROCESSOR_FILE, "wb") as f:
            cloudpickle.dump(preprocessor, f)

        postprocessor = prepare_data_template.get_postprocessor()
        with open(MAIN_POSTPROCESSOR_FILE, "wb") as f:
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

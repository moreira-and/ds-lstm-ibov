from config import logger
from config.paths import TRAIN_PARAMS_FILE
from shared import ConfigWrapper

from typing import Optional
import numpy as np
import pandas as pd

import inspect

from keras.utils import timeseries_dataset_from_array

def sliding_window_generator(data: pd.DataFrame):
    try:
        params = ConfigWrapper(config_path=TRAIN_PARAMS_FILE)
        targets = params.get("targets")

        if targets is None or not isinstance(targets, list):
            logger.error("Config 'targets' must be uma lista com nomes de colunas.")
            raise ValueError("Invalid targets configuration.")

        sequence_length = params.get("sequence_length")
        batch_size = params.get("batch_size")

        logger.info(f"Generating Timeseries: len={len(data)}, sequence_length={sequence_length}, batch_size={batch_size}")

        data = data.apply(pd.to_numeric, errors='coerce').astype("float32")

        features = data
        labels = data[targets]

        return timeseries_dataset_from_array(
            data=features,
            targets=labels,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=False
        )
    except Exception:
        logger.exception(f"Error in {inspect.currentframe().f_code.co_name}")
        raise




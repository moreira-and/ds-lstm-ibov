from typing import Optional
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from config import logger
from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper


def sliding_window_generator(
    data: pd.DataFrame
) -> TimeseriesGenerator:

    params = ConfigWrapper(config_path=TRAIN_PARAMS_FILE)
    targets = params.get("targets")

    if targets is None or not isinstance(targets, list):
        raise ValueError("Config 'targets' must be a non-empty list of column names.")

    sequence_length = params.get("sequence_length")
    batch_size = params.get("batch_size")

    # Extrai os valores das colunas alvo
    target_values = data[targets].values
    data_values = data.values  # todas as colunas como input

    logger.info(f"Generating Timeseries: len={len(data)}, sequence_length={sequence_length}, batch_size={batch_size}")

    return TimeseriesGenerator(
        data=data_values,
        targets=target_values,
        length=sequence_length,
        batch_size=batch_size,
        shuffle=False
    )


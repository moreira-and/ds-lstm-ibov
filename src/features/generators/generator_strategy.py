from abc import ABC, abstractmethod
from src.config.config import logger

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DefaultRnnGenerator(IGeneratorStrategy):
    def __init__(self, sequence_length=7, batch_size=1):
        self._sequence_length = sequence_length
        self._batch_size = batch_size

    def generate(self, data, targets=None):

        logger.info("Generating Timeseries from dataset...")

        y = targets if targets is not None and len(targets) > 0 else data

        return TimeseriesGenerator(
            data = data,
            targets = y,
            length=self._sequence_length,
            batch_size=self._batch_size,
            shuffle=False
        )

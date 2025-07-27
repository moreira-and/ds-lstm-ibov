from config import logger
from config.paths import TRAIN_PARAMS_FILE
from ..interfaces import IGeneratorStrategy

from utils import ConfigWrapper

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

class SlidingWindowGenerator(IGeneratorStrategy):
    def __init__(self):
        params = ConfigWrapper(config_path=TRAIN_PARAMS_FILE)
        self._sequence_length = params.get("sequence_length")
        self._batch_size = params.get("batch_size")

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
from abc import ABC, abstractmethod

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

class GeneratorBuilder(ABC):
    @abstractmethod
    def generate(self, data):
        pass

class TimeseriesGeneratorBuilder(GeneratorBuilder):
    def __init__(self, sequence_length=7, batch_size=1):
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def generate(self, data):
        return TimeseriesGenerator(
            data, data,
            length=self.sequence_length,
            batch_size=self.batch_size
        )
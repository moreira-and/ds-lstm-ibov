from abc import ABC, abstractmethod
from typing import Union, Optional, List

import pandas as pd
import numpy as np

class ITrainStrategy(ABC):
    @abstractmethod
    def train(self, model, train_gen):
        pass
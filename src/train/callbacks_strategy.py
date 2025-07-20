
from abc import ABC, abstractmethod
from typing import List
from keras import callbacks


class ICallbacksStrategy(ABC):
    @staticmethod
    @abstractmethod
    def get() -> List:
        raise NotImplementedError("Implement in subclass")


class RegressionCallbacksStrategy(ICallbacksStrategy):
    @staticmethod
    def get():
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True,mode='min', verbose=0)
        lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=16, min_lr=1e-5,mode='min', verbose=0)
        return [early_stop,lr_schedule]

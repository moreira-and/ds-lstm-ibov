
from abc import ABC, abstractmethod
from typing import List
from keras import callbacks


class CallbacksStrategy(ABC):
    @staticmethod
    @abstractmethod
    def get() -> List:
        pass


class RegressionCallbacksStrategy(CallbacksStrategy):
    @staticmethod
    def get():
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True,mode='min', verbose=0)
        lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6,mode='min', verbose=0)
        return [early_stop,lr_schedule]

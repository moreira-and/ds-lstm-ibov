
from abc import ABC, abstractmethod
from typing import List
from keras import callbacks


class CallbacksStrategy(ABC):
    @staticmethod
    @abstractmethod
    def get() -> List:
        pass


class DefaultCallbacksStrategy(CallbacksStrategy):
    @staticmethod
    def get():
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        return [early_stop,lr_schedule]

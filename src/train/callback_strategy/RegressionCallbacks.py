from src.train.interfaces import ICallbacksStrategy
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

class RegressionCallbacks(ICallbacksStrategy):
    @staticmethod
    def get():
        early_stop = EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True,mode='min', verbose=0)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=16, min_lr=1e-5,mode='min', verbose=0)
        return [early_stop,lr_schedule]

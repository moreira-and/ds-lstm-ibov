from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper

from keras.callbacks import EarlyStopping,ReduceLROnPlateau

def RegressionCallbacks():
        params = ConfigWrapper(TRAIN_PARAMS_FILE)
        patience = int(params.get("patience"))

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True,mode='min', verbose=0)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(patience/2), min_lr=1e-5,mode='min', verbose=0)
        return [early_stop,lr_schedule]

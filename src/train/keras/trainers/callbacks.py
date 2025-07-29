from config.paths import TRAIN_PARAMS_FILE
from utils import ConfigWrapper

from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def regression_callbacks():
    params = ConfigWrapper(TRAIN_PARAMS_FILE)

    patience = int(params.get("patience"))
    monitor = params.get("monitor", "val_loss")
    mode = params.get("mode", "min")

    reduce_lr_patience = int(params.get("reduce_lr_patience", patience // 2))
    reduce_lr_factor = float(params.get("reduce_lr_factor", 0.5))
    min_lr = float(params.get("min_lr", 1e-5))

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        mode=mode,
        verbose=0
    )

    lr_schedule = ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        mode=mode,
        verbose=0
    )

    return [early_stop, lr_schedule]

from config import logger
from config.paths import TRAIN_PARAMS_FILE
from shared import ConfigWrapper

from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def regression_callbacks():
    try:
        params = ConfigWrapper(TRAIN_PARAMS_FILE)

        monitor = params.get("monitor")
        mode = params.get("mode")

        early_stop_patience = params.get("early_stop_patience")
        reduce_lr_patience = params.get("reduce_lr_patience")
        reduce_lr_factor = params.get("reduce_lr_factor")
        min_lr = params.get("min_lr")

        logger.info(f"Creating callbacks with patience={early_stop_patience}, monitor='{monitor}', mode='{mode}', "
                    f"reduce_lr_patience={reduce_lr_patience}, reduce_lr_factor={reduce_lr_factor}, min_lr={min_lr}")


        early_stop = EarlyStopping(
            monitor=monitor,
            patience=early_stop_patience,
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

    except Exception as e:
        logger.exception(f"Failed to create {function.__name__}")
        raise

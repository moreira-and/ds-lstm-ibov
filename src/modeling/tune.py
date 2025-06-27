from pathlib import Path
import time

import joblib
from tqdm import tqdm
import typer

import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


from src.config import MODELS_DIR, PROCESSED_DATA_DIR, logger

from src.utils.train.compile_strategy import RegressionCompileStrategy
from src.utils.train.callbacks_strategy import RegressionCallbacksStrategy
from src.utils.train.metric_strategy import RegressionMetricStrategy

from src.utils.tune.tune_template import TunerKerasPipeline
from src.utils.tune.tuner_builder import RegressionRobustModelTuner
from src.utils.tune.search_strategy import RegressionTuneStrategy

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    X_path: Path = PROCESSED_DATA_DIR / "X_train.npy",
    y_path: Path = PROCESSED_DATA_DIR / "y_train.npy",
    # -----------------------------------------
    epochs: int = 2**8,
    batch_size: int = 2**6,
    validation_len: int = 2**5,
    # -----------------------------------------
    experiment_name: str = "default_experiment",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    start_time = time.time()
    logger.info("Loading training dataset...")

    X_train = np.load(X_path)
    y_train = np.load(y_path)

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]
    
    logger.info("Selecting builder strategy...")
    tuner_builder = RegressionRobustModelTuner(
        input_shape=input_shape
        ,output_shape=output_shape
        ,max_trials = 14
        ,project_name = "default"
        ,compile_strategy = RegressionCompileStrategy(
                optimizer = Adam(learning_rate=0.001)
                ,loss = Huber(delta=1.0)
                ,metrics = RegressionMetricStrategy().get_metrics()
            )
    )

    logger.info("Selecting Tuning strategy...")
    searcher = RegressionTuneStrategy(
        batch_size=batch_size
        ,epochs=epochs
        ,validation_len=validation_len
        ,callbacks=RegressionCallbacksStrategy.get()
    )

    logger.info("Building model training pipeline template...")   
    template = TunerKerasPipeline(
        tuner_builder = tuner_builder
        ,searcher = searcher
    )

    logger.info("Tuning model...")

    best_model, best_hps = template.run(X_train,y_train)

    print("\nMelhores Hiperparâmetros Encontrados:")
    print("-" * 40)
    for param in best_hps.values.keys():
        print(f"{param:<20}: {best_hps.get(param)}")
    print("-" * 40)

    model_name = 'best_model_tuned.keras'

    

    logger.info(f"Saving '{model_name}' in '{MODELS_DIR}'...")

    best_model.save(MODELS_DIR / model_name)

    logger.success("Modeling training complete.")
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    '''
    logger.info("Logging experiment into mlflow.")

    ml_logger = MLflowLogger(
        model=best_model,
        history=history,
        validation_len=validation_len,
        batch_size=batch_size,
        X_train=X_train,
        y_train=y_train,
        elapsed_time=elapsed_time
    )

    ml_logger.log_run(run_name=best_model.__class__.__name__,
                      experiment_name=experiment_name)

    logger.success("Experiment logged successfully.")
    '''

    # -----------------------------------------


if __name__ == "__main__":
    app()

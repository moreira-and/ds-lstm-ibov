from pathlib import Path
import time

import joblib
from tqdm import tqdm
import typer

import numpy as np

import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
#tf.compat.v1.reset_default_graph instead

from src.config.config import MODELS_DIR, PROCESSED_DATA_DIR, logger

#from src.utils.train.compile_strategy import RegressionCompileStrategy
#from src.utils.train.callbacks_strategy import RegressionCallbacksStrategy
#from src.utils.train.metric_strategy import RegressionMetricStrategy

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
    epochs: int = 256,    
    validation_len: int = 128,
    batch_size: int = 64,
    # -----------------------------------------
    experiment_name: str = "default_experiment",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    start_time = time.time()
    logger.info("Loading training dataset...")

    X_train = np.nan_to_num(np.load(X_path), nan=0)
    y_train = np.nan_to_num(np.load(y_path), nan=0)

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]
    
    logger.info("Selecting builder strategy...")
    tuner_builder = RegressionRobustModelTuner(
        input_shape=input_shape
        ,output_shape=output_shape
        ,max_trials = 128
        ,project_name = "default"
    )

    logger.info("Selecting Tuning strategy...")
    searcher = RegressionTuneStrategy(
        batch_size=batch_size
        ,epochs=epochs
        ,validation_len=validation_len
        #,callbacks=RegressionCallbacksStrategy.get()
    )

    logger.info("Building model training pipeline template...")   
    template = TunerKerasPipeline(
        tuner_builder = tuner_builder
        ,searcher = searcher
    )

    assert tf.executing_eagerly(), "TensorFlow is not executing eagerly!"

    logger.info("Tuning model...")

    best_model, best_hps = template.run(X_train,y_train)

    if best_hps.values.keys() is not None:
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

    # -----------------------------------------


if __name__ == "__main__":
    app()

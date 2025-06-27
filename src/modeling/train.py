from pathlib import Path
import time

import joblib
from tqdm import tqdm
import typer

import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


from src.config import MODELS_DIR, PROCESSED_DATA_DIR, logger
from src.utils.train.metric_strategy import RegressionMetricStrategy
from src.utils.train.callbacks_strategy import RegressionCallbacksStrategy

from src.utils.train.model_template import ModelKerasPipeline
from src.utils.train.model_builder import RegressionRobustModelBuilder
from src.utils.train.compile_strategy import RegressionCompileStrategy
from src.utils.train.train_strategy import RegressionTrainStrategy
from src.utils.train.logger_strategy import MLflowLogger

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_path: Path = None,
    X_path: Path = PROCESSED_DATA_DIR / "X_train.npy",
    y_path: Path = PROCESSED_DATA_DIR / "y_train.npy",
    # -----------------------------------------
    optimizer: str = None,
    loss: str = None,
    metrics: str = None,
    # -----------------------------------------
    batch_size: int = 128,
    epochs: int = 30,
    validation_len: int = 45,
    # -----------------------------------------
    experiment_name: str = "default_experiment",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()
    logger.info("Loading training dataset...")

    X_train = np.load(X_path)
    y_train = np.load(y_path)

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]
    
    logger.info("Selecting builder strategy...")
    model_builder = RegressionRobustModelBuilder(
        input_shape=input_shape,
        output_shape=output_shape
    )
    
    logger.info("Selecting compile strategy...")
    compiler = RegressionCompileStrategy(
        optimizer = Adam(learning_rate=0.001) if optimizer is None else optimizer, 
        loss = Huber(delta=1.0) if loss is None else loss, 
        metrics = RegressionMetricStrategy().get_metrics()  if metrics is None else metrics
    )

    logger.info("Selecting training strategy...")
    trainer = RegressionTrainStrategy(
        batch_size=batch_size
        ,epochs=epochs
        ,validation_len=validation_len
        ,callbacks=RegressionCallbacksStrategy.get()
    )

    logger.info("Building model training pipeline template...")   
    template = ModelKerasPipeline(
        model_builder=model_builder,
        compiler=compiler,
        trainer=trainer
    )

    logger.info("Training model...")

    model, history = template.run(X_train,y_train)
        
    final_epoch = history.epoch[-1]
    final_loss = history.history['loss'][-1]

    model_name = f'{model.__class__.__name__}_epoch{final_epoch}_loss{final_loss:.4f}.keras'

    logger.info(f"Saving '{model_name}' in '{MODELS_DIR}'...")

    model.save(MODELS_DIR / model_name)

    logger.success("Modeling training complete.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    logger.info("Logging experiment into mlflow.")

    ml_logger = MLflowLogger(
        model=model,
        history=history,
        validation_len=validation_len,
        batch_size=batch_size,
        X_train=X_train,
        y_train=y_train,
        elapsed_time=elapsed_time
    )

    ml_logger.log_run(run_name=model.__class__.__name__,
                      experiment_name=experiment_name)

    logger.success("Experiment logged successfully.")

    # -----------------------------------------


if __name__ == "__main__":
    app()

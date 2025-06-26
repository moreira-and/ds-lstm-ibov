from pathlib import Path
import time

import joblib
#from loguru import logger
from tqdm import tqdm
import typer

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


from src.utils.train.metric_strategy import ClassificationMetricStrategy, RegressionMetricStrategy,smape, rmse, r2_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, logger
from src.utils.train.model_template import ModelKerasPipeline
from src.utils.train.model_builder import RegressionRobustModelBuilder,RegressionSimpleModelBuilder
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
    epochs: int = 300,
    validation_len: int = 30,
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

    if model_path is None:
        logger.info("No model path provided. Building a new model from scratch...")
        model = RegressionRobustModelBuilder(
            input_shape=input_shape,
            output_shape=output_shape
        ).build_model()
    else:
        logger.info(f"Loading model from {model_path}...")
        model = load_model(
            model_path,
            custom_objects={
                "smape": smape,
                "rmse": rmse,
                "r2_score": r2_score
            }
        )

    logger.info("Selecting compile strategy...")
    compiler = RegressionCompileStrategy(
        optimizer = Adam(learning_rate=0.001) if optimizer is None else optimizer, 
        loss = Huber(delta=1.0) if loss is None else loss, 
        metrics = RegressionMetricStrategy().get_metrics()  if metrics is None else metrics
    )

    logger.info("Selecting training strategy...")
    trainer = RegressionTrainStrategy(
        batch_size=batch_size,
        epochs=epochs,
        validation_len=validation_len,
        callbacks=None
    )

    logger.info("Building model training pipeline template...")   
    template = ModelKerasPipeline(
        model=model,
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

    ml_logger = MLflowLogger(
        model=model,
        history=history,
        model_strategy=model.__class__.__name__,
        compile_strategy=compiler.__class__.__name__,
        train_strategy=trainer.__class__.__name__,
        batch_size=batch_size,
        input_shape=input_shape,
        output_shape=output_shape,
        experiment_name=experiment_name,
        model_version="v1.0.0",
    )

    ml_logger.log_run(run_name=model.__class__.__name__)

    logger.success("Modeling training complete.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()

from pathlib import Path
import time

import joblib
from tqdm import tqdm
import typer

import numpy as np

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, logger

from src.utils.train.callbacks_strategy import RegressionCallbacksStrategy

from src.utils.train.model_template import ModelKerasPipeline
from src.utils.train.model_builder import RegressionRobustModelBuilder,LoadKerasModelBuilder
from src.utils.train.compile_strategy import RegressionCompileStrategy
from src.utils.train.train_strategy import RegressionTrainStrategy

from src.utils.log.log_strategy import KerasExperimentMlFlowLogger

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    X_path: Path = PROCESSED_DATA_DIR / "X_train.npy",
    y_path: Path = PROCESSED_DATA_DIR / "y_train.npy",
    # -----------------------------------------
    epochs: int = 256,    
    validation_len: int = 64,
    batch_size: int = 32,
    # -----------------------------------------
    experiment_name: str = "default_experiment",
    model_name: str = "default_model",
    # -----------------------------------------
    model_path: Path = None, # MODELS_DIR / "default_model.keras", 
    # -----------------------------------------
    optimizer: str = None,
    loss: str = None,
    metrics: str = None,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()
    logger.info("Loading training dataset...")

    X_train = np.load(X_path)
    y_train = np.load(y_path)

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]
    logger.info(f"Input shape: {input_shape}, Output shape: {output_shape}")
    
    logger.info("Selecting builder strategy...")
    if model_path:
        model_builder = LoadKerasModelBuilder(model_path=model_path)
        logger.info(f"Loading model from {model_path}")
    else:
        logger.info("Building new model...")
        model_builder = RegressionRobustModelBuilder(
            input_shape=input_shape,
            output_shape=output_shape
        )
    
    logger.info("Selecting compile strategy...")
    compiler = RegressionCompileStrategy()

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
    logger.info("Model training complete.")

    model_name = f'{model_name}.keras'

    logger.info(f"Saving '{model_name}' in '{MODELS_DIR}'...")

    model.save(MODELS_DIR / model_name)

    logger.success("Modeling training complete.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    logger.info("Logging experiment into mlflow.")

    ml_logger = KerasExperimentMlFlowLogger(
        model=model,
        history=history,
        validation_len=validation_len,
        batch_size=batch_size,
        elapsed_time=elapsed_time
    )

    ml_logger.run(
        run_name="training_run",
        experiment_name=experiment_name,
        model_name="regression-pipeline",        
        purpose_tag = "regression-pipeline")

    logger.success("Experiment logged successfully.")

    # -----------------------------------------


if __name__ == "__main__":
    app()

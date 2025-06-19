from pathlib import Path
import time

import joblib
from loguru import logger
from tqdm import tqdm
import typer

import numpy as np

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.train.model_template import ModelKerasPipeline
from src.utils.train.model_builder import RegressionRobustModelBuilder
from src.utils.train.compile_strategy import RegressionCompileStrategy
from src.utils.train.train_strategy import RegressionTrainStrategy

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    X_path: Path = PROCESSED_DATA_DIR / "X_train.npy",
    y_path: Path = PROCESSED_DATA_DIR / "y_train.npy"
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()
    logger.info("Loading training dataset...")

    X_train = np.load(X_path)
    y_train = np.load(y_path)

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]

    logger.info("Building model...")
    builder = RegressionRobustModelBuilder(
            input_shape = input_shape,
            output_shape = output_shape
        )

    logger.info("Selecting compile strategy...")
    compiler = RegressionCompileStrategy()

    logger.info("Selecting training strategy...")
    trainer = RegressionTrainStrategy(batch_size=3, epochs=1000, validation_split=0.15)

    logger.info("Building model training pipeline template...")   
    template = ModelKerasPipeline(
        builder=builder,
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
    # -----------------------------------------


if __name__ == "__main__":
    app()

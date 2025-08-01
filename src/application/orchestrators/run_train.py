from pathlib import Path
import time

import joblib
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd

from config import logger
from config.paths import MODELS_DIR, MAIN_RAW_FILE, TRAIN_PARAMS_FILE

from models.keras.runners import TrainerKerasRunner
from models.keras.compilers import RegressionCompile
from models.keras.model_builders import ModelLoader, RegressionSequentialRobust
from models.keras.trainers import RegressionForecast

from log.runners import PipelineRunner
from log.loggers.mlflow import (
    LogDictArtifacts, LogDictParameters, LogDictTags,
    LogPythonModel, LogTrainHistoryMetrics,
    LogValPredictionsPlot, LogTestPredictionsPlot
)

from utils import ConfigWrapper

app = typer.Typer()


@app.command()
def main(

    # -----------------------------------------
    experiment_name: str = "default_experiment",
    run_name: str ="training_run",
    model_name: str = "default_model",
    # -----------------------------------------
    model_path: Path = None, # MODELS_DIR / "default_model.keras", 
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    start_time = time.time()
    logger.info("Loading training dataset...")
    df = pd.read_csv(MAIN_RAW_FILE, index_col=0).sort_index()

    logger.info("Loading params...")
    params = ConfigWrapper(TRAIN_PARAMS_FILE)    
    sequence_length = params.get("sequence_length")
    targets = params.get("targets")
    
    logger.info("Selecting builder strategy...")
    if model_path:
        model_builder = ModelLoader(model_path=model_path)
        logger.info(f"Loading model from {model_path}")
    else:
        logger.info("Building new model...")
        model_builder = RegressionSequentialRobust(
            input_shape=(sequence_length, df.shape[1]),
            output_shape=(len(targets), )
        )
    
    logger.info("Selecting compile strategy...")
    compiler = RegressionCompile()

    logger.info("Selecting training strategy...")
    trainer = RegressionForecast()

    logger.info("Building model training pipeline template...")   
    runner = TrainerKerasRunner(
        model_builder=model_builder,
        compiler=compiler,
        trainer=trainer
    )

    logger.info("Training model...")
    model, history = runner.fit(df)
    logger.info("Model training complete.")

    model_name = f'{model_name}.keras'

    logger.info(f"Saving '{model_name}' in '{MODELS_DIR}'...")

    model.save(MODELS_DIR / model_name)

    logger.success("Modeling training complete.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    logger.info("Logging experiment into mlflow.")

    ml_logger = PipelineRunner(
        [
            LogDictArtifacts(),
            LogDictParameters(),
            LogDictTags(),
            LogPythonModel(),
            LogTrainHistoryMetrics(),
            LogValPredictionsPlot(),
            LogTestPredictionsPlot()
        ]
    )

    ml_logger.run(
        experiment_name= experiment_name,
        run_name= run_name,            
        model_name= model_name
    )

    logger.success("Experiment logged successfully.")

    # -----------------------------------------


if __name__ == "__main__":
    app()

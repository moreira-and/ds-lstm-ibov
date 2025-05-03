from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.train.model_template import ModelKerasPipeline
from src.utils.train.model_builder import LstmModelBuilder
from src.utils.train.compile_strategy import ClassificationCompileStrategy
from src.utils.train.train_strategy import BasicTrainStrategy

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")

    X_train = np.load(PROCESSED_DATA_DIR / X_train)
    y_train = np.load(PROCESSED_DATA_DIR / y_train)

    builder = LstmModelBuilder()

    compiler = ClassificationCompileStrategy()

    trainer = BasicTrainStrategy()

    template = ModelKerasPipeline(
        builder=builder,
        compiler=compiler,
        trainer=trainer
    )

    model = template.run(X_train,y_train)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

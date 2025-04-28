from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR
from src.utils.prepare_strategy import ModelHandler


import pandas as pd

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    
    dataset = pd.read_csv(input_path, index_col=0)

    handler = ModelHandler(dataset)

    

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

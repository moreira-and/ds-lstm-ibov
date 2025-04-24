from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, CONFIG_DIR
from src.utils.dataset_loader import MultiLoader

# from scikitlearn import split

app = typer.Typer()


@app.command()
def main(
    # ----  DEFAULT PATHS --------------------------
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    train_path: Path = PROCESSED_DATA_DIR / "dataset_train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "dataset_test.csv",
    # ----------------------------------------------
):
    # -----------------------------------------
    logger.info("Starting dataset loading...")
    dataset = MultiLoader.load()
    # logger.info("Splitting dataset into training and test sets...")
    # X,y = slipt...
    # logger.success("Dataset successfully loaded.")  # se estiver usando `loguru`

    # -----------------------------------------


if __name__ == "__main__":
    app()

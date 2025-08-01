from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

import mlflow

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

CONFIG_DIR = PROJ_ROOT / "configs"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTED_DATA_DIR = DATA_DIR / "predicted"

MODELS_DIR = PROJ_ROOT / "models"

MLFLOW_TRACKING_URI = PROJ_ROOT / "mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

REPORTS_DIR = PROJ_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

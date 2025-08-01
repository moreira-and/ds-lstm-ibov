from .dotenv_loader import load_env
from .paths import *
from .mlflow_config import mlflow_tracking

from .logging_config import setup_logging
from loguru import logger as _logger
logger = _logger

load_env()
setup_logging()
mlflow_tracking()


__all__ = ["logger"]

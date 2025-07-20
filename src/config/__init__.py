from .dotenv_loader import load_env
from .paths import *
from .mlflow_config import configure_mlflow

from .logging_config import setup_logging
from loguru import logger as _logger
logger = _logger

load_env()
setup_logging()
configure_mlflow()


__all__ = ["logger"]

import yaml
from pathlib import Path
from typing import Dict, Optional

from config.paths import DATASET_PARAMS_FILE
from utils.config_wrapper import ConfigWrapper

class MarketConfig:
    def __init__(self, config: Optional[Dict] = None, path: Path = DATASET_PARAMS_FILE):
        self._config = config or ConfigWrapper(path).config

    @property
    def tickers(self) -> Dict[str, str]:
        return self._config.get("yfinance", "tickers_code")

    @property
    def sgs_codes(self) -> Dict[str, int]:
        return self._config.get("bcb", "sgs_code")

    @property
    def datareader_codes(self) -> Dict[str, str]:
        return self._config.get("DataReader", "reader_code")
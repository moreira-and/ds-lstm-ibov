from pathlib import Path
from typing import Dict, Optional

from src.utils.dataset.config_wrapper import ConfigWrapper
from src.config.config import CONFIG_DIR

class MarketConfigFacade:
    def __init__(self, config: Optional[Dict] = None, path: Path = CONFIG_DIR / 'dataset.yaml'):
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
from typing import Dict
from src.utils.config_loader import ConfigLoader
from src.config import CONFIG_DIR

class MarketConfigLoader:
    def __init__(self):
        self.loader = ConfigLoader(CONFIG_DIR / 'dataset.yaml')

    @property
    def tickers(self) -> Dict[str, str]:
        return self.loader.get("yfinance", "tickers_code")

    @property
    def sgs_codes(self) -> Dict[str, int]:
        return self.loader.get("bcb", "sgs_code")

    @property
    def datareader_codes(self) -> Dict[str, str]:
        return self.loader.get("DataReader", "reader_code")
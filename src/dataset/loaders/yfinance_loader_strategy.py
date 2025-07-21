from config import logger
from dataset.interfaces import IDatasetLoader
from dataset.utils import MarketConfig

from typing import Dict

import time
import yfinance as yf
import pandas as pd

class YfinanceLoader(IDatasetLoader):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfig().tickers

    def load(self) -> Dict[str, pd.DataFrame]:
        try:
            return self._load_from_yfinance()
        except Exception as e:
            logger.error(f'Error loading data in {self.__class__.__name__}: {e}')
            return {}

    def _load_from_yfinance(self,interval="1d") -> Dict[str, pd.DataFrame]:
        yf_data = {}
        for name, ticker in self._config.items():
            logger.info(f'Downloading {name} ({ticker}) from yfinance...')
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date,auto_adjust=True,interval=interval)
                if not df.empty:
                    yf_data[name] = df
                else:
                    logger.warning(f'No data returned for {ticker}')
            except Exception as e:
                logger.error(f'Error loading {ticker}: {e}')
            time.sleep(2)
            
        return yf_data
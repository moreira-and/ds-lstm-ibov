from config import logger
from dataset.helpers import MarketConfig
from dataset.interfaces import IDatasetLoaderStrategy

from typing import Dict, Optional
import time
import pandas as pd

from pandas_datareader import data as pdr


class DataReaderLoader(IDatasetLoaderStrategy):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfig().datareader_codes

    def load(self) -> Dict[str, pd.DataFrame]:
        dr_data = {}

        for ticker, name in self._config.items():
            logger.info(f'Downloading {name} ({ticker}) from DataReader...')
            df = self._load_single_ticker(ticker, name)
            if df is not None:
                dr_data[name] = df

            time.sleep(2)

        return dr_data

    def _load_single_ticker(self, ticker: str, name: str) -> Optional[pd.DataFrame]:
        try:
            df = pdr.DataReader(ticker, 'fred', start=self.start_date, end=self.end_date)
            if df.empty:
                logger.warning(f"No data returned for {name} ({ticker})")
                return None
            return df
        except Exception as e:
            logger.error(f"Error loading {name} ({ticker}): {e}", exc_info=True)
            return None
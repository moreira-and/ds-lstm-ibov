
from abc import ABC, abstractmethod

import time
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import CONFIG_DIR, logger

from src.utils.dataset.market_config_facade import MarketConfigFacade

import yfinance as yf
from pandas_datareader import data as pdr
import requests

class IDatasetLoadingStrategy(ABC):
    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError("Implement in subclass")


class DatasetMultiLoader(IDatasetLoadingStrategy):
    def __init__(self, strategies: List[IDatasetLoadingStrategy]):
        self.strategies = strategies

    def load(self) -> Dict[str, pd.DataFrame]:
            
        dataset = {}

        for strategy in self.strategies:
            name = strategy.__class__.__name__.replace("LoadingStrategy","")
            try:
                dataset[name] = strategy.load()
            except Exception as e:
                logger.error(f"Error loading data with {name}: {e}")
                pass
        return dataset


class YfinanceLoadingStrategy(IDatasetLoadingStrategy):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfigFacade().tickers

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


    
class BcbLoadingStrategy(IDatasetLoadingStrategy):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = pd.to_datetime(start_date, dayfirst=True)
        self.end_date = pd.to_datetime(end_date, dayfirst=True)
        self._config = MarketConfigFacade().sgs_codes

    def load(self) -> Dict[str, pd.DataFrame]:
        bcb_data = {}

        for name, ticker in self._config.items():
            logger.info(f"Downloading {name} ({ticker}) from the Central Bank of Brazil API...")
            df = self._load_single_ticker(name, ticker)
            if df is not None:
                bcb_data[name] = df

            time.sleep(2)

        return bcb_data

    def _load_single_ticker(self, name: str, ticker: str) -> Optional[pd.DataFrame]:
        try:
            data = self._request_bcb_series(ticker)
            if not data:
                logger.warning(f"No data returned for {name} ({ticker})")
                return None

            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df.set_index('data', inplace=True)
            return df.rename(columns={"valor": name})
        except Exception as e:
            logger.error(f"Error processing {name} ({ticker}): {e}", exc_info=True)
            return None

    def _request_bcb_series(self, sgs_code: str) -> Optional[Dict]:
        url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sgs_code}/dados'
        params = {
            'formato': 'json',
            'dataInicial': self.start_date.strftime('%d/%m/%Y'),
            'dataFinal': self.end_date.strftime('%d/%m/%Y'),
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Erro ao conectar à API do BCB: {e}", exc_info=True)
            return None
        except ValueError:
            logger.error("Erro ao interpretar a resposta como JSON.", exc_info=True)
            return None



class DataReaderLoadingStrategy(IDatasetLoadingStrategy):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfigFacade().datareader_codes

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
from loguru import logger
import requests
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.config import CONFIG_DIR

from src.utils.market_config_loader import MarketConfigLoader
from src.utils.config_loader import ConfigLoader

import yfinance as yf

# Caminho do seu arquivo
yaml_path = CONFIG_DIR / 'dataset.yaml'

class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Any:
        """
        Método obrigatório que todas as classes que herdam devem implementar.
        Deve retornar um dataset (ex: DataFrame, lista de dicionários, etc.).
        """
        pass


class MultiLoader(DatasetLoader):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def load(self) -> Dict[str, pd.DataFrame]:
        try:
            dataset = {}
            dataset['Yfinance'] = YfinanceLoader(self.start_date,self.end_date).load()
            dataset['Bcb'] =  BcbLoader(self.start_date,self.end_date).load()
            dataset['PandasReader'] = DataReaderLoader(self.start_date,self.end_date).load()
            return dataset
        
        except Exception as e:
            logger.error(f'Error loading data in {self.__class__}: {e}')


class YfinanceLoader(DatasetLoader):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfigLoader().tickers

    def load(self) -> Dict[str, pd.DataFrame]:
        try:
            yf_data = {}

            for nome, ticker in self._config.items():
                print(f'Downloading {nome} ({ticker}) from yfinance...')

                try:
                    df_ticker = yf.download(ticker, start=self.start_date, end=self.end_date)
                    if (not df_ticker.empty):
                        yf_data[nome] = df_ticker
                    else:
                        print(f'Warning: No data returned for {ticker}')
                except Exception as e:
                    print(f'Error loading {ticker}: {e}')

            return yf_data
        except Exception as e:
            logger.error(f'Error loading data in {self.__class__}: {e}')
            pass
    

    
class BcbLoader(DatasetLoader):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfigLoader().sgs_codes

    def load(self) -> Dict[str, pd.DataFrame]:
        try:
            bcb_data = {}
            for name, ticker  in self._config.items():
                print(f"Downloading {name} ({ticker}) from the Central Bank of Brazil API...")
                try:
                    df_ticker = pd.DataFrame(self._request_bcb_series(ticker))
                    if not df_ticker.empty:
                        df_ticker['data'] = pd.to_datetime(df_ticker['data'])
                        df_ticker.set_index('data',inplace=True)
                        bcb_data[name] = df_ticker.rename(columns={"valor": name})
                    else:
                        print(f'Warning: No data returned for {ticker}')
                except Exception as e:
                    print(f'Error loading {ticker}: {e}')
                    pass

            return bcb_data
        except Exception as e:
            logger.error(f'Error loading data in {self.__class__}: {e}')
    
    def _request_bcb_series(self, sgs_code):
        url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sgs_code}/dados'
        params = {
            'formato': 'json',
            'dataInicial': pd.to_datetime(self.start_date, dayfirst=True, format = '%d/%m/%Y').strftime('%d/%m/%Y'),
            'dataFinal': pd.to_datetime(self.end_date, dayfirst=True,format = '%d/%m/%Y').strftime('%d/%m/%Y'),
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                print(f"Warning: No data found for SGS code {sgs_code} between {self.start_date} and {self.end_date}.")
                return []
            return data
        except requests.exceptions.RequestException as e:
            print(f"Erro ao conectar à API do BCB: {e}")
            return []
        except ValueError:
            print("Erro ao interpretar a resposta como JSON.")
            return []


from pandas_datareader import data as pdr

class DataReaderLoader(DatasetLoader):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._config = MarketConfigLoader().datareader_codes

    def load(self) -> Dict[str, pd.DataFrame]:
        dr_data = {}

        for ticker, name in self._config.items():
            print(f'Downloading {name} ({ticker}) from DataReader...')
            try:
                df_ticker =  pdr.DataReader(ticker, 'fred', start=self.start_date, end=self.end_date)
                if not df_ticker.empty:
                    dr_data[name] = df_ticker
                else:
                    print(f'Warning: No data returned for {ticker}')
            except Exception as e:
                print(f'Error loading {ticker}: {e}')

        return dr_data
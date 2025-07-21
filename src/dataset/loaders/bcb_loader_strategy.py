
from config import logger
from dataset.interfaces import IDatasetLoader
from dataset.utils import MarketConfig

from typing import Dict, Optional

import requests
import time
import pandas as pd

class BcbLoader(IDatasetLoader):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = pd.to_datetime(start_date, dayfirst=True)
        self.end_date = pd.to_datetime(end_date, dayfirst=True)
        self._config = MarketConfig().sgs_codes

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
import pytest
import yaml
from src.utils.config_loader import ConfigLoader


@pytest.fixture
def sample_config_dict():
    return {
        "yfinance": {
            "tickers_code": {
                "BOVESPA": "^BVSP",
                "BITCOIN": "BTC-USD"
            }
        },
        "bcb": {
            "sgs_code": {
                "SELIC": 11
            }
        }
    }


@pytest.fixture
def sample_config_file(tmp_path, sample_config_dict):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_file


def test_load_config_success(sample_config_file):
    loader = ConfigLoader(str(sample_config_file))
    assert loader.config["yfinance"]["tickers_code"]["BOVESPA"] == "^BVSP"


def test_get_nested_value(sample_config_file):
    loader = ConfigLoader(str(sample_config_file))
    result = loader.get("yfinance", "tickers_code", "BITCOIN")
    assert result == "BTC-USD"


def test_get_with_default(sample_config_file):
    loader = ConfigLoader(str(sample_config_file))
    result = loader.get("nonexistent", default="not_found")
    assert result == "not_found"


def test_missing_file_raises_error(tmp_path):
    fake_path = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        ConfigLoader(str(fake_path))



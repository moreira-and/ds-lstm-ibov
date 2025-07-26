import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigWrapper:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        self.config = self.__load()

    def __load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")
        with open(self.path, "r") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str) -> Any:
        data = self.config
        for key in keys:
            data = data.get(key, {})
        return data
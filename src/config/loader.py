import os
import yaml
from typing import Any, Dict


class ConfigLoader:

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"File not found at: {self.config_path}")
        self.config_data = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_path,"r") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise RuntimeError(f"Error parsing YAML configuration: {exc}")

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self.config_data
        try:
            for key in keys:
                value = value[key]
            return value
            
        except KeyError:
            return default


if __name__=="__main__":
    loader= ConfigLoader()
    print(f"Project Name:{loader.get("project.name")}")





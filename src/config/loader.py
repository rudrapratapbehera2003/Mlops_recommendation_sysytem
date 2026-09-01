import os
import re
import yaml
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class ConfigLoader:

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"File not found at: {self.config_path}")
        self.config_data = self._load_yaml()

    def _resolve_env_placeholders(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._resolve_env_placeholders(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_env_placeholders(item) for item in value]
        if not isinstance(value, str):
            return value

        pattern = r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}"

        def replace(match):
            env_name = match.group(1)
            default = match.group(2)
            return os.getenv(env_name, default if default is not None else "")

        return re.sub(pattern, replace, value)

    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as file:
            try:
                loaded = yaml.safe_load(file) or {}
                return self._resolve_env_placeholders(loaded)
            except yaml.YAMLError as exc:
                raise RuntimeError(f"Error parsing YAML configuration: {exc}")

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self.config_data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


if __name__ == "__main__":
    loader = ConfigLoader()
    print(f"Project Name: {loader.get('project.name')}")


"""Preferencias del usuario (idioma, etc.) en la carpeta de datos de la app."""
import os
from pathlib import Path

import yaml

from fenotit import APP_NAME


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / APP_NAME if base else Path.home() / f".{APP_NAME.lower()}"


def _file() -> Path:
    return data_dir() / "settings.yaml"


def load() -> dict:
    try:
        with open(_file(), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get(key: str, default=None):
    return load().get(key, default)


def set(key: str, value) -> None:
    data = load()
    data[key] = value
    try:
        _file().parent.mkdir(parents=True, exist_ok=True)
        with open(_file(), "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
    except OSError:
        pass

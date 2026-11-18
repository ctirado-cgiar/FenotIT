"""Preferencias del usuario (idioma, etc.) en la carpeta de datos de la app."""
import os
from pathlib import Path

import yaml

# APP_NAME es el nombre del programa.
from fenotit import APP_NAME


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / APP_NAME if base else Path.home() / f".{APP_NAME.lower()}"


def _file() -> Path:
    return data_dir() / "settings.yaml"


def load() -> dict:
    try:
        with open(_file(), encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get(key: str, default=None):
    return load().get(key, default)


def set(key: str, value) -> None:
    data = load()
    data[key] = value
    try:
        _file().parent.mkdir(parents=True, exist_ok=True)
        with open(_file(), "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, allow_unicode=True)
    except OSError:
        pass


def recent_projects() -> list[str]:
    """Proyectos abiertos o guardados hace poco (los que todavía existen)."""
    return [path for path in get("recent_projects", []) or [] if Path(path).exists()]


def recent_presets() -> list[str]:
    """Presets cargados o guardados hace poco (los que todavía existen)."""
    return [path for path in get("recent_presets", []) or [] if Path(path).exists()]


def add_recent_preset(path) -> None:
    path = str(Path(path).resolve())
    items = [path] + [item for item in get("recent_presets", []) or [] if item != path]
    set("recent_presets", items[:8])


def add_recent_project(path) -> None:
    path = str(Path(path).resolve())
    items = [path] + [item for item in get("recent_projects", []) or [] if item != path]
    set("recent_projects", items[:8])

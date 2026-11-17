"""Registro de mensajes en un archivo de la carpeta de datos de la aplicación."""
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fenotit import APP_NAME, __version__


def log_dir() -> Path:
    from fenotit.settings import data_dir
    return data_dir() / "logs"


def log_file() -> Path:
    return log_dir() / "fenotit.log"


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"fenotit.{name}")


def setup(console_level=logging.WARNING) -> Path | None:
    root = logging.getLogger("fenotit")
    if root.handlers:
        return log_file()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)

    path = None
    try:
        log_dir().mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_file(), maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(handler)
        path = log_file()
    except OSError as e:
        root.warning("No se pudo crear el log en %s: %s", log_dir(), e)

    def _hook(exc_type, exc, trace):
        root.critical("Error no controlado", exc_info=(exc_type, exc, trace))
    sys.excepthook = _hook
    threading.excepthook = lambda hook_args: root.critical(
        "Error no controlado en hilo %s", hook_args.thread.name if hook_args.thread else "?",
        exc_info=(hook_args.exc_type, hook_args.exc_value, hook_args.exc_traceback))

    root.info("%s %s | Python %s | %s", APP_NAME, __version__, sys.version.split()[0], sys.platform)
    return path

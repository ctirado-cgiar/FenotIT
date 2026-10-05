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
        fh = RotatingFileHandler(log_file(), maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)
        path = log_file()
    except OSError as e:
        root.warning("No se pudo crear el log en %s: %s", log_dir(), e)

    def _hook(exc_type, exc, tb):
        root.critical("Error no controlado", exc_info=(exc_type, exc, tb))
    sys.excepthook = _hook
    threading.excepthook = lambda a: root.critical(
        "Error no controlado en hilo %s", a.thread.name if a.thread else "?",
        exc_info=(a.exc_type, a.exc_value, a.exc_traceback))

    root.info("%s %s | Python %s | %s", APP_NAME, __version__, sys.version.split()[0], sys.platform)
    return path

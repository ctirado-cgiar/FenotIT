"""Conversor de modelos instalado aparte: un Python privado en la carpeta de datos
(`converter/`) con PyTorch para CPU, Ultralytics y onnx, que usa core/segmentation/convert.py
para pasar un .pt a ONNX. Desde el código fuente se crea con venv; en la aplicación instalada
(Windows) se descargan el Python embebible oficial (python.org) y pip. Se descarga solo
cuando el usuario lo pide y no forma parte de FenotIT (Ultralytics tiene licencia AGPL-3.0)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Callable

PY_VERSION = "3.12.10"
EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_CPU = "https://download.pytorch.org/whl/cpu"
PACKAGES = ["ultralytics", "onnx"]
SIZE_GB = 1.5                                      # espacio aproximado con PyTorch para CPU


class Cancelled(Exception):
    pass


def env_dir() -> Path:
    from fenotit import settings
    return settings.data_dir() / "converter"


def env_python() -> Path:
    folder = env_dir()
    if sys.platform.startswith("win"):
        embedded = folder / "python.exe"
        return embedded if embedded.exists() else folder / "Scripts" / "python.exe"
    return folder / "bin" / "python"


def installed() -> bool:
    return env_python().exists() and (env_dir() / "ready").exists()


def size_mb() -> float:
    folder = env_dir()
    if not folder.exists():
        return 0.0
    return sum(file.stat().st_size for file in folder.rglob("*") if file.is_file()) / 1e6


def uninstall():
    shutil.rmtree(env_dir(), ignore_errors=True)


def _run(cmd: list[str], log: Callable[[str], None], cancel: threading.Event):
    """Corre un comando y pasa cada línea de su salida a `log`; Cancelled si se cancela."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            creationflags=flags)
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            lines.append(line)
            log(line)
        if cancel.is_set():
            proc.terminate()
            proc.wait()
            raise Cancelled()
    if proc.wait() != 0:
        raise RuntimeError("\n".join(lines[-4:]) or f"exit code {proc.returncode}")


def _download(url: str, dest: Path, log: Callable[[str], None], cancel: threading.Event):
    import urllib.request
    log(url)
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out_file:
        while chunk := response.read(1 << 16):
            if cancel.is_set():
                raise Cancelled()
            out_file.write(chunk)


def _embedded_python(log, cancel):
    """Python embebible de python.org con pip (Windows, aplicación instalada)."""
    folder = env_dir()
    zip_path = folder / "python.zip"
    _download(EMBED_URL, zip_path, log, cancel)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(folder)
    zip_path.unlink()
    for pth in folder.glob("python*._pth"):              # habilita site-packages
        text = pth.read_text().replace("#import site", "import site")
        if "Lib\\site-packages" not in text:
            text += "\nLib\\site-packages\n"
        pth.write_text(text)
    get_pip = folder / "get-pip.py"
    _download(GET_PIP_URL, get_pip, log, cancel)
    _run([str(env_python()), str(get_pip), "--no-warn-script-location"], log, cancel)
    get_pip.unlink()


def install(log: Callable[[str], None] = print, cancel: threading.Event | None = None):
    """Crea el Python privado e instala PyTorch para CPU (si ese índice no responde, el de
    PyPI), Ultralytics y onnx. Al terminar deja la marca `ready`."""
    cancel = cancel or threading.Event()
    folder = env_dir()
    if folder.exists() and not installed():
        shutil.rmtree(folder, ignore_errors=True)        # instalación anterior incompleta
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if not env_python().exists():
            if getattr(sys, "frozen", False):
                if not sys.platform.startswith("win"):
                    raise RuntimeError("automatic install is only available on Windows")
                _embedded_python(log, cancel)
            else:
                _run([sys.executable, "-m", "venv", str(folder)], log, cancel)
        interpreter = str(env_python())
        pip = [interpreter, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location"]
        _run([interpreter, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"], log, cancel)
        try:
            _run(pip + ["torch", "torchvision", "--index-url", TORCH_CPU], log, cancel)
        except RuntimeError:
            log("PyPI")
            _run(pip + ["torch", "torchvision"], log, cancel)
        _run(pip + PACKAGES, log, cancel)
        (folder / "ready").write_text("ok", encoding="utf-8")
    except Cancelled:
        shutil.rmtree(folder, ignore_errors=True)
        raise

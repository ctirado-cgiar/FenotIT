"""Modelos de IA disponibles para segmentar: los de `models/` junto a la aplicación y los que
agrega el usuario (carpeta de datos `models/`). Cada modelo es una carpeta con `model.yaml`:

    name: MobileSAM
    output: masks            # masks = contorno de cada objeto; boxes = caja de cada objeto
    format: sam              # sam, yolo o detr
    files: [mobilesam_encoder.onnx, mobilesam_decoder.onnx]
    classes: []              # nombres de las clases (modelos de cajas)
    input: 640               # lado de la imagen que recibe el modelo (modelos de cajas)
    license: Apache-2.0
    source: https://github.com/ChaoningZhang/MobileSAM

`add(ruta)` copia un detector .onnx a la carpeta de datos y escribe su `model.yaml` con lo
que se lee del archivo (formato, clases, tamaño de entrada)."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import yaml

OUTPUTS = ("masks", "boxes")


def roots() -> list[Path]:
    """Carpetas `models/` donde se buscan modelos, de la aplicación primero."""
    from fenotit import settings
    here = Path(__file__).resolve()
    frozen = getattr(sys, "frozen", False)
    out = [Path(getattr(sys, "_MEIPASS", "")) if frozen else None, Path(sys.executable).parent if frozen else None,
           here.parents[3], settings.data_dir()]
    return [r / "models" for r in out if r is not None]


def user_root() -> Path:
    from fenotit import settings
    return settings.data_dir() / "models"


def _read(folder: Path) -> dict | None:
    f = folder / "model.yaml"
    if not f.exists():
        return None
    try:
        info = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    files = [str(x) for x in info.get("files") or []]
    if not files or not all((folder / x).exists() for x in files) or info.get("output") not in OUTPUTS:
        return None
    return {"id": folder.name, "dir": str(folder), "name": str(info.get("name") or folder.name),
            "output": info["output"], "format": str(info.get("format", "")), "files": files,
            "classes": [str(c) for c in info.get("classes") or []], "input": int(info.get("input") or 640),
            "license": str(info.get("license", "")), "source": str(info.get("source", ""))}


def available() -> list[dict]:
    """Modelos que se pueden usar (con onnxruntime instalado), sin repetir carpeta."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return []
    seen, out = set(), []
    for root in roots():
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            info = _read(folder)
            if info and info["id"] not in seen:
                seen.add(info["id"])
                out.append(info)
    return out


def get(model_id: str) -> dict | None:
    return next((m for m in available() if m["id"] == model_id), None)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "model"


def _names_from_yaml(onnx_path: Path) -> list[str]:
    """Clases de un `data.yaml` junto al modelo (exportaciones de Roboflow y Ultralytics)."""
    for f in (onnx_path.with_name("data.yaml"), onnx_path.parent.parent / "data.yaml"):
        if f.exists():
            try:
                names = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("names")
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names)]
            if isinstance(names, list):
                return [str(n) for n in names]
    return []


def add(path: str, name: str | None = None) -> dict:
    """Agrega un detector .onnx: lee su formato y clases, lo copia a la carpeta de datos y
    devuelve su descripción. ValueError("pt") si es un .pt de PyTorch; ValueError("format")
    si la salida del modelo no es de un formato conocido."""
    from fenotit.core.segmentation import detect
    src = Path(path)
    if src.suffix.lower() in (".pt", ".pth"):
        raise ValueError("pt")
    info = detect.inspect(str(src))
    names = info["classes"] or _names_from_yaml(src)
    n = info["n_classes"] or len(names)
    if info["format"] == "detr" and names and n == len(names) + 1:
        n -= 1                                # índice 0 reservado (exportaciones de Roboflow)
    if len(names) != n:
        names = (names + [f"class_{i + 1}" for i in range(len(names), n)])[:n]
    title = name or src.stem
    folder = user_root() / _slug(title)
    k = 2
    while folder.exists():
        folder = user_root() / f"{_slug(title)}-{k}"
        k += 1
    folder.mkdir(parents=True)
    shutil.copy2(src, folder / "model.onnx")
    meta = {"name": title, "output": "boxes", "format": info["format"], "files": ["model.onnx"],
            "classes": names, "input": info["input"], "source": str(src)}
    (folder / "model.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _read(folder)


def remove(model_id: str) -> bool:
    """Borra un modelo agregado por el usuario (los de la aplicación no se borran)."""
    folder = user_root() / model_id
    if folder.is_dir() and (folder / "model.yaml").exists():
        shutil.rmtree(folder)
        return True
    return False

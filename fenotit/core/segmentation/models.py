"""Modelos de IA disponibles para segmentar: los de `models/` junto a la aplicación y los que
agrega el usuario (carpeta de datos `models/`). Cada modelo es una carpeta con `model.yaml`:

    name: MobileSAM
    output: masks            # masks = contorno de cada objeto; boxes = caja de cada objeto
    format: sam              # sam, yolo o detr
    files: [mobilesam_encoder.onnx, mobilesam_decoder.onnx]
    classes: []              # nombres de las clases (modelos de cajas)
    input: 640               # lado de la imagen que recibe el modelo (modelos de cajas)
    description: …           # para qué sirve: qué objetos, cultivo, tipo de foto
    owner: …                 # autor o institución dueña del modelo
    license: Apache-2.0
    source: https://github.com/ChaoningZhang/MobileSAM   # dónde se obtuvo
    citation: …              # cómo citarlo
    added: 2026-11-09        # fecha en que se agregó (modelos del usuario)
    file: …                  # archivo original que agregó el usuario

`add(ruta)` copia un detector (.onnx, o .pt convertido a ONNX) a la carpeta de datos y
escribe su `model.yaml` con lo que se lee del archivo (formato, clases, tamaño de entrada);
`update` cambia el nombre, las clases o la ficha (descripción, propietario, licencia,
fuente, cita). Los modelos agregados quedan en la carpeta de datos y siguen disponibles en
las sesiones siguientes."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import yaml

OUTPUTS = ("masks", "boxes")
INFO = ("description", "owner", "license", "source", "citation")   # ficha del modelo (propiedad intelectual)


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
            **{k: str(info.get(k) or "") for k in INFO}, "added": str(info.get("added") or ""),
            "file": str(info.get("file") or ""), "tool": str(info.get("tool") or "")}


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


def add(path: str, name: str | None = None, classes: list[str] | None = None) -> dict:
    """Agrega un detector: un .onnx, o un .pt/.pth que se convierte a ONNX
    (core/segmentation/convert.py). Lee el formato y las clases (del modelo, de un
    `data.yaml` junto al archivo o de `classes`), lo copia a la carpeta de datos y devuelve
    su descripción. ValueError("pt") si es un .pt y no hay herramienta para convertirlo;
    ValueError("format") si la salida del modelo no es de un formato conocido."""
    import tempfile
    from fenotit.core.segmentation import convert, detect
    src = original = Path(path)
    converted = {}
    if src.suffix.lower() in (".pt", ".pth"):
        if not convert.tools():
            raise ValueError("pt")
        converted = convert.to_onnx(str(src), tempfile.mkdtemp(prefix="fenotit_onnx_"))
        src = Path(converted["onnx"])
    info = detect.inspect(str(src))
    if converted.get("input"):
        info["input"] = int(converted["input"])
    names = list(classes or []) or info["classes"] or list(converted.get("classes") or []) \
        or _names_from_yaml(original)
    n = info["n_classes"] or len(names)
    if info["format"] == "detr" and names and n == len(names) + 1:
        n -= 1                                # índice 0 reservado (exportaciones de Roboflow)
    if len(names) != n:
        names = (names + [f"class_{i + 1}" for i in range(len(names), n)])[:n]
    title = name or original.stem
    folder = user_root() / _slug(title)
    k = 2
    while folder.exists():
        folder = user_root() / f"{_slug(title)}-{k}"
        k += 1
    folder.mkdir(parents=True)
    shutil.copy2(src, folder / "model.onnx")
    import datetime
    found = info.get("meta") or {}
    meta = {"name": title, "output": "boxes", "format": info["format"], "files": ["model.onnx"],
            "classes": names, "input": info["input"],
            "description": found.get("description", ""), "owner": found.get("author", ""),
            "license": found.get("license", ""), "source": found.get("docs", ""), "citation": "",
            "added": datetime.date.today().isoformat(), "file": str(original),
            "tool": converted.get("tool", "")}
    (folder / "model.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _read(folder)


def generic_names(model: dict) -> bool:
    """Las clases no tienen nombre propio (class_1, class_2…)."""
    return all(c == f"class_{i + 1}" for i, c in enumerate(model["classes"]))


def update(model_id: str, **changes) -> dict:
    """Cambia el nombre, las clases (mismo número que da el modelo) o la ficha (INFO) de un
    modelo agregado por el usuario."""
    folder = user_root() / model_id
    f = folder / "model.yaml"
    meta = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    if "classes" in changes:
        names = [str(n).strip() for n in changes["classes"]]
        if len(names) != len(meta.get("classes") or []):
            raise ValueError("count")
        meta["classes"] = names
    for key in ("name",) + INFO:
        if key in changes:
            meta[key] = str(changes[key]).strip()
    f.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _read(folder)


def is_user(model: dict) -> bool:
    """El modelo lo agregó el usuario (se puede editar y quitar)."""
    return Path(model["dir"]).parent.resolve() == user_root().resolve()


def missing_info(model: dict) -> list[str]:
    """Campos obligatorios de la ficha que están vacíos (descripción, propietario, licencia)."""
    return [k for k in ("description", "owner", "license") if not model.get(k)]


def remove(model_id: str) -> bool:
    """Borra un modelo agregado por el usuario (los de la aplicación no se borran)."""
    folder = user_root() / model_id
    if folder.is_dir() and (folder / "model.yaml").exists():
        shutil.rmtree(folder)
        return True
    return False

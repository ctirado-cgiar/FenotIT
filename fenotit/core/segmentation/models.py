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

`add(ruta)` guarda un detector (.onnx, .pt convertido a ONNX o paquete .fntm) en la
carpeta de datos, cifrado para que solo lo abra FenotIT (core/segmentation/vault.py), y
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
PACKAGE = ".fntm"                                  # paquete para compartir un modelo entre equipos con FenotIT
OLD_PACKAGES = (".fenotitmodel",)                  # extensión de versiones anteriores: se lee igual
ENCRYPTED_FILE = "model.enc"                       # el modelo cifrado dentro de su carpeta y del paquete
INFO = ("description", "owner", "license", "source", "citation")   # ficha del modelo (propiedad intelectual)


def roots() -> list[Path]:
    """Carpetas `models/` donde se buscan modelos, de la aplicación primero."""
    from fenotit import settings
    here = Path(__file__).resolve()
    frozen = getattr(sys, "frozen", False)
    out = [Path(getattr(sys, "_MEIPASS", "")) if frozen else None, Path(sys.executable).parent if frozen else None,
           here.parents[3], settings.data_dir()]
    return [root / "models" for root in out if root is not None]


def user_root() -> Path:
    from fenotit import settings
    return settings.data_dir() / "models"


def _read(folder: Path) -> dict | None:
    yaml_file = folder / "model.yaml"
    if not yaml_file.exists():
        return None
    try:
        info = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    files = [str(x) for x in info.get("files") or []]
    if not files or not all((folder / x).exists() for x in files) or info.get("output") not in OUTPUTS:
        return None
    return {"id": folder.name, "dir": str(folder), "name": str(info.get("name") or folder.name),
            "output": info["output"], "format": str(info.get("format", "")), "files": files,
            "classes": [str(class_name) for class_name in info.get("classes") or []], "input": int(info.get("input") or 640),
            **{key: str(info.get(key) or "") for key in INFO}, "added": str(info.get("added") or ""),
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
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            info = _read(folder)
            if info and info["id"] not in seen:
                seen.add(info["id"])
                out.append(info)
    return out


def get(model_id: str) -> dict | None:
    return next((model for model in available() if model["id"] == model_id), None)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "model"


def _names_from_yaml(onnx_path: Path) -> list[str]:
    """Clases de un `data.yaml` junto al modelo (exportaciones de Roboflow y Ultralytics)."""
    for candidate in (onnx_path.with_name("data.yaml"), onnx_path.parent.parent / "data.yaml"):
        if candidate.exists():
            try:
                names = (yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}).get("names")
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(names, dict):
                return [str(names[key]) for key in sorted(names)]
            if isinstance(names, list):
                return [str(name) for name in names]
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
    if src.suffix.lower() in (PACKAGE, *OLD_PACKAGES):
        return import_package(str(src))
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
    class_count = info["n_classes"] or len(names)
    if info["format"] == "detr" and names and class_count == len(names) + 1:
        class_count -= 1                                # índice 0 reservado (exportaciones de Roboflow)
    if len(names) != class_count:
        names = (names + [f"class_{i + 1}" for i in range(len(names), class_count)])[:class_count]
    title = name or original.stem
    folder = user_root() / _slug(title)
    number = 2
    while folder.exists():
        folder = user_root() / f"{_slug(title)}-{number}"
        number += 1
    folder.mkdir(parents=True)
    from fenotit.core.segmentation import vault
    (folder / ENCRYPTED_FILE).write_bytes(vault.lock(src.read_bytes()))     # solo lo abre FenotIT
    import datetime
    found = info.get("meta") or {}
    meta = {"name": title, "output": "boxes", "format": info["format"], "files": [ENCRYPTED_FILE],
            "classes": names, "input": info["input"],
            "description": found.get("description", ""), "owner": found.get("author", ""),
            "license": found.get("license", ""), "source": found.get("docs", ""), "citation": "",
            "added": datetime.date.today().isoformat(), "file": str(original),
            "tool": converted.get("tool", "")}
    (folder / "model.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if converted:                                     # el .onnx convertido no queda en el disco
        shutil.rmtree(src.parent, ignore_errors=True)
    return _read(folder)


def export_package(model_id: str, dest: str) -> str:
    """Guarda un modelo agregado por el usuario como paquete `.fntm` (su model.yaml y
    el modelo cifrado) para usarlo en otro equipo con FenotIT."""
    import zipfile
    folder = user_root() / model_id
    out = Path(dest)
    if out.suffix.lower() != PACKAGE:
        out = out.with_name(out.name + PACKAGE)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for file_name in ["model.yaml"] + list((_read(folder) or {}).get("files") or []):
            z.write(folder / file_name, file_name)
    return str(out)


def import_package(path: str) -> dict:
    """Agrega un modelo desde un paquete `.fntm`. ValueError("package") si no es un
    paquete de FenotIT o si se cifró con otra clave."""
    import zipfile
    from fenotit.core.segmentation import vault
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise ValueError("package")
    with z:
        names = set(z.namelist())
        if "model.yaml" not in names:
            raise ValueError("package")
        meta = yaml.safe_load(z.read("model.yaml")) or {}
        files = [str(file_name) for file_name in meta.get("files") or []]
        if not files or any(file_name not in names or "/" in file_name or "\\" in file_name or file_name.startswith(".") for file_name in files):
            raise ValueError("package")
        try:
            vault.unlock(z.read(files[0]))
        except ValueError:
            raise ValueError("package")
        title = str(meta.get("name") or Path(path).stem)
        folder = user_root() / _slug(title)
        number = 2
        while folder.exists():
            folder = user_root() / f"{_slug(title)}-{number}"
            number += 1
        folder.mkdir(parents=True)
        for file_name in ["model.yaml"] + files:
            (folder / file_name).write_bytes(z.read(file_name))
    return _read(folder)


def generic_names(model: dict) -> bool:
    """Las clases no tienen nombre propio (class_1, class_2…)."""
    return all(class_name == f"class_{i + 1}" for i, class_name in enumerate(model["classes"]))


def update(model_id: str, **changes) -> dict:
    """Cambia el nombre, las clases (mismo número que da el modelo) o la ficha (INFO) de un
    modelo agregado por el usuario."""
    folder = user_root() / model_id
    yaml_file = folder / "model.yaml"
    meta = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    if "classes" in changes:
        names = [str(name).strip() for name in changes["classes"]]
        if len(names) != len(meta.get("classes") or []):
            raise ValueError("count")
        meta["classes"] = names
    for key in ("name",) + INFO:
        if key in changes:
            meta[key] = str(changes[key]).strip()
    yaml_file.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _read(folder)


def is_user(model: dict) -> bool:
    """El modelo lo agregó el usuario (se puede editar y quitar)."""
    return Path(model["dir"]).parent.resolve() == user_root().resolve()


def missing_info(model: dict) -> list[str]:
    """Campos obligatorios de la ficha que están vacíos (descripción, propietario, licencia)."""
    return [key for key in ("description", "owner", "license") if not model.get(key)]


def remove(model_id: str) -> bool:
    """Borra un modelo agregado por el usuario (los de la aplicación no se borran)."""
    folder = user_root() / model_id
    if folder.is_dir() and (folder / "model.yaml").exists():
        shutil.rmtree(folder)
        return True
    return False

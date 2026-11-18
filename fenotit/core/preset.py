"""Preset: la configuración de un montaje de captura (correcciones, escala, áreas, segmentación,
análisis y sus parámetros) en un archivo .fntp (YAML), para aplicarla a otro proyecto.
De los modelos de IA guarda solo cuál se usó."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import yaml

# Versión del programa, que queda escrita en el preset.
from fenotit import __version__
# Las correcciones de imagen, que el preset guarda y vuelve a aplicar.
from fenotit.core.corrections.pipeline import Corrections
# El proyecto y sus partes (escala, segmentación) que el preset lee y modifica.
from fenotit.core.project import Project, Scale, Segmentation
# from_legacy lee las áreas guardadas en el formato de versiones anteriores.
from fenotit.core.roi import from_legacy

PRESET_EXT = ".fntp"
OLD_PRESET_EXTS = (".fenotitpreset",)     # extensión de versiones anteriores: se lee igual
FORMAT = 1
PARTS = ("corrections", "scale", "areas", "segmentation", "analysis")
OWN_OF_A_PHOTO = ("scale", "areas")           # partes que una foto puede tener distintas a las demás
PART_FIELDS = {"corrections": ("corrections", "max_mpx"), "scale": ("scale", "unit"), "areas": ("areas",),
               "segmentation": ("segmentation",), "analysis": ("analysis", "params")}


def applicable(project: Project, path=None) -> tuple:
    """Partes que el proyecto tiene configuradas y un preset puede llevar. `path`: la foto
    abierta (sus áreas y su escala cuentan)."""
    from fenotit.core.analysis.registry import ANALYSES
    analysis = ANALYSES.get(project.analysis)
    has = {"corrections": bool(project.corrections.active()),
           "scale": project.scale_for(path).mm_per_pixel is not None,
           "areas": bool(project.roi_for(path)),
           "segmentation": analysis is None or bool(analysis.segmentation),
           "analysis": analysis is not None}
    return tuple(part for part in PARTS if has[part])


def suggested(project: Project, path=None) -> tuple:
    """Partes aplicables que valen para todas las fotos: las áreas o la escala propias de la
    foto abierta quedan fuera."""
    own = {"scale": bool(path) and project.key(path) in project.image_scale,
           "areas": project.has_own_roi(path)}
    return tuple(part for part in applicable(project, path) if not own.get(part))


def cautions(project: Project, path=None) -> dict:
    """Parte → motivo por el que conviene revisarla antes de guardarla: "own_areas" y
    "own_scale" si son solo de la foto abierta, "fixed_card" si la tarjeta de color depende
    de que el montaje no cambie (lugar fijo o foto de referencia)."""
    found = {}
    if project.has_own_roi(path):
        found["areas"] = "own_areas"
    if path and project.key(path) in project.image_scale:
        found["scale"] = "own_scale"
    card = project.corrections.color
    if card.enabled and (card.mode == "fixed" or card.reference == "photo"):
        found["corrections"] = "fixed_card"
    return found


def from_project(project: Project, name: str, parts=PARTS, description: str = "",
                 areas: list | None = None, scale: Scale | None = None) -> dict:
    """El preset con las partes pedidas del proyecto. `areas` y `scale`: los que se guardan
    (por defecto los de todas las fotos)."""
    data = {"fenotit_preset": FORMAT, "fenotit_version": __version__, "name": name, "description": description}
    if "corrections" in parts:
        corrections = project.corrections.to_dict()
        corrections["distortion"]["photos"] = []           # fotos del tablero: son de ese proyecto
        data["corrections"] = corrections
        data["max_mpx"] = project.max_mpx
    if "scale" in parts:
        data["scale"] = asdict(project.scale if scale is None else scale)
        data["unit"] = project.unit
    if "areas" in parts:
        data["areas"] = copy.deepcopy(project.roi if areas is None else areas)
    if "segmentation" in parts:
        data["segmentation"] = asdict(project.segmentation)
    if "analysis" in parts and project.analysis:
        data["analysis"] = project.analysis
        data["params"] = copy.deepcopy(project.params.get(project.analysis, {}))
    return data


def parts_of(data: dict) -> tuple:
    """Partes que trae el preset, en el orden de PARTS."""
    return tuple(part for part in PARTS if PART_FIELDS[part][0] in data)


def save(data: dict, path) -> Path:
    path = Path(path)
    if path.suffix != PRESET_EXT:
        path = path.with_name(path.name + PRESET_EXT)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return path


def load(path) -> dict:
    """Lee un preset; ValueError("not_a_preset") si el archivo no lo es."""
    with open(path, encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError:
            data = None
    if not isinstance(data, dict) or "fenotit_preset" not in data:
        raise ValueError("not_a_preset")
    data.setdefault("name", Path(path).stem)
    return data


def apply(data: dict, project: Project) -> tuple:
    """Pone en el proyecto las partes que trae el preset; lo demás queda como estaba.
    Devuelve las partes aplicadas."""
    parts = parts_of(data)
    if "corrections" in parts:
        photos = project.corrections.distortion.photos
        project.corrections = Corrections.from_dict(copy.deepcopy(data["corrections"]))
        project.corrections.distortion.photos = photos
        project.max_mpx = data.get("max_mpx", project.max_mpx)
        perspective = project.corrections.perspective
        if perspective.enabled and perspective.width_mm and perspective.height_mm:
            project.apply_scale_to_all(Scale(None, "aruco"))       # la escala sale de los marcadores de cada foto
    if "scale" in parts:
        project.apply_scale_to_all(Scale(**data["scale"]))
        project.unit = data.get("unit", project.unit)
    if "areas" in parts:
        project.apply_roi_to_all(from_legacy(copy.deepcopy(data["areas"])))
    if "segmentation" in parts:
        project.segmentation = Segmentation.from_dict(copy.deepcopy(data["segmentation"]))
    if "analysis" in parts:
        project.analysis = data["analysis"]
        project.params[data["analysis"]] = copy.deepcopy(data.get("params") or {})
    project.preset = {"name": data.get("name", ""), "data": copy.deepcopy(data)}
    return parts


def rebase(project: Project):
    """Toma el estado actual del proyecto como el del preset aplicado (después de que la
    interfaz completa los valores que el archivo no traía)."""
    applied = (project.preset or {}).get("data")
    if applied:
        project.preset["data"] = from_project(project, applied.get("name", ""), parts_of(applied),
                                              applied.get("description", ""))


def modified(project: Project) -> bool:
    """El proyecto ya no coincide con el preset que se le aplicó (en las partes que trae)."""
    applied = (project.preset or {}).get("data")
    if not applied:
        return False
    current = from_project(project, applied.get("name", ""), parts_of(applied))
    return any(_comparable(field, current.get(field)) != _comparable(field, applied.get(field))
               for part in parts_of(applied) for field in PART_FIELDS[part]
               if field in current or field in applied)


def _comparable(field: str, value):
    """El valor sin lo que el programa calcula solo: con umbral automático, el rango Mín/Máx
    sale de cada foto."""
    value = _plain(value)
    if field == "segmentation" and isinstance(value, dict) and value.get("auto"):
        value = {key: item for key, item in value.items() if key not in ("min_val", "max_val")}
    return value


def _plain(value):
    """El valor como queda después de pasar por el archivo (tuplas = listas)."""
    return yaml.safe_load(yaml.safe_dump(value)) if value is not None else None

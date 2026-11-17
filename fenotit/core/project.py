"""Estado de una sesión de trabajo, guardable como carpeta con <carpeta>.fnt (YAML)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from fenotit import __version__
from fenotit.core.corrections.pipeline import Corrections
from fenotit.core.roi import from_legacy

PREVIEW = "preview.jpg"
PROJECT_EXT = ".fnt"
OLD_PROJECT_EXTS = (".fenotit",)       # extensiones de versiones anteriores: se leen y al guardar pasan a la actual
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Segmentation:
    """Paso 1 (core.segmentation). Las clases y el detalle son de la segmentación por
    ejemplos; los trazos de cada foto van en Project.strokes."""
    method: str = "threshold"     # threshold | classifier | model
    color_space: str = "BGR"
    channel: int = 0
    min_val: int = 0
    max_val: int = 255
    auto: bool = False            # umbral automático (Otsu) en el canal elegido
    refine: list = field(default_factory=lambda: [{"op": "fill_holes"}])   # afinar máscara, en orden
    classes: list = field(default_factory=lambda: [dict(entry) for entry in DEFAULT_CLASSES])
    detail: float = 1.0           # MP de la copia en que se clasifica (por ejemplos)
    model: dict = field(default_factory=lambda: {"name": "mobilesam", "mode": "auto", "grid": 16})
    detector: dict = field(default_factory=dict)   # detector de cajas (análisis Detección y detecciones por objeto)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Segmentation":
        data = dict(data or {})
        operation = data.pop("morph_op", "none")                     # versiones anteriores: una operación
        size, repeats = data.pop("morph_size", 3), data.pop("morph_iter", 1)
        if "refine" not in data:
            data["refine"] = ([{"op": operation, "size": int(size), "iter": int(repeats)}] if operation != "none" else []) \
                + [{"op": "fill_holes"}]
        model = data.get("model") or {}
        if "detector" not in data and model.get("output") == "boxes":   # detector guardado como modelo de Segmentar
            data["detector"] = {key: value for key, value in model.items() if key not in ("output", "mode", "grid")}
            data["model"] = {"name": "mobilesam", "mode": "auto", "grid": 16}
            if data.get("method") == "model":
                data["method"] = "threshold"
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})

    def object_classes(self) -> list[int]:
        return [i for i, entry in enumerate(self.classes) if i and entry.get("object", True)]


DEFAULT_CLASSES = [{"name": "Fondo", "color": "#9E9E9E", "object": False},
                   {"name": "Objeto", "color": "#2166AC", "object": True}]


@dataclass
class Scale:
    mm_per_pixel: float | None = None
    source: str = "none"          # none | manual | two_points | aruco
    detail: str = ""


@dataclass
class Project:
    name: str = ""
    folder: Path | None = None
    images: list[Path] = field(default_factory=list)
    current_index: int = 0
    mode: str = "individual"      # individual | batch
    analysis: str | None = None   # id del módulo (ej. "morphometry")
    segmentation: Segmentation = field(default_factory=Segmentation)
    scale: Scale = field(default_factory=Scale)
    unit: str = "mm"              # unidad de los resultados: la que se usó al fijar la escala
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    roi: list[dict] = field(default_factory=list)          # áreas de todas las fotos (core.roi, 0-1)
    image_roi: dict[str, list] = field(default_factory=dict)     # foto -> sus propias áreas
    image_scale: dict[str, Scale] = field(default_factory=dict)  # foto -> su propia escala
    excluded: dict[str, list] = field(default_factory=dict)      # foto -> objetos excluidos a mano ([x, y] 0-1)
    highlighted: dict[str, list] = field(default_factory=dict)   # foto -> objetos destacados ★ ([x, y] 0-1)
    strokes: dict[str, list] = field(default_factory=dict)       # foto -> trazos de ejemplo (segmentación por ejemplos)
    prompts: dict[str, list] = field(default_factory=dict)       # foto -> clics para el modelo de IA ([x, y, objeto?] 0-1)
    regions: dict[str, dict] = field(default_factory=dict)       # foto -> categorías de las regiones del modelo
                                                                 # {"default": cat, "picks": [[x, y, cat], ...]}
    corrections: Corrections = field(default_factory=Corrections)
    metadata: dict[str, str] = field(default_factory=dict)   # {"file": ruta, "key_column": col}
    max_mpx: float | None = 50.0                             # resolución de trabajo del análisis (None = completa)
    ran: list[str] = field(default_factory=list)             # análisis ya corridos (para ofrecer recalcular al abrir)
    preset: dict = field(default_factory=dict)               # preset aplicado: {"name", "data"} (core.preset)
    display: dict[str, Any] = field(default_factory=lambda: {"legend": True, "color_format": "RGB",
                                                             "legend_scale": 1.0})

    @property
    def current_image(self) -> Path | None:
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]
        return None

    # ── Ajustes por foto: los propios de la foto o, si no tiene, los de todas ──

    @staticmethod
    def key(path) -> str:
        return Path(path).as_posix()

    def roi_for(self, path) -> list[dict]:
        return self.image_roi.get(self.key(path), self.roi) if path else self.roi

    def has_own_roi(self, path) -> bool:
        return bool(path) and self.key(path) in self.image_roi

    def set_roi(self, path, shapes: list[dict]):
        self.image_roi[self.key(path)] = list(shapes)

    def apply_roi_to_all(self, shapes: list[dict]):
        self.roi = list(shapes)
        self.image_roi = {}

    def others_with_own_roi(self, path) -> int:
        return sum(1 for photo in self.image_roi if photo != self.key(path))

    def marks_for(self, kind: str, path) -> list[list[float]]:
        """Objetos marcados a mano en una foto: kind = "excluded" | "highlighted"."""
        return list(getattr(self, kind).get(self.key(path), [])) if path else []

    def set_marks(self, kind: str, path, points: list):
        store = getattr(self, kind)
        if points:
            store[self.key(path)] = [[round(float(x), 6), round(float(y), 6)] for x, y in points]
        else:
            store.pop(self.key(path), None)

    def excluded_for(self, path) -> list[list[float]]:
        return self.marks_for("excluded", path)

    def set_excluded(self, path, points: list):
        self.set_marks("excluded", path, points)

    def scale_for(self, path) -> Scale:
        return self.image_scale.get(self.key(path), self.scale) if path else self.scale

    def set_scale(self, path, scale: Scale):
        self.image_scale[self.key(path)] = scale

    def apply_scale_to_all(self, scale: Scale):
        self.scale = scale
        self.image_scale = {}

    def others_with_own_scale(self, path) -> int:
        return sum(1 for photo in self.image_scale if photo != self.key(path))

    @property
    def preview(self) -> Path | None:
        """Miniatura del último resultado (la muestran los recientes de la pantalla de inicio)."""
        return self.folder / PREVIEW if self.folder else None

    @property
    def file(self) -> Path | None:
        return self.folder / f"{self.folder.name}{PROJECT_EXT}" if self.folder else None

    def set_images(self, paths, mode: str | None = None):
        self.images = [Path(path) for path in paths]
        self.current_index = 0
        if mode:
            self.mode = mode

    def _rel(self, path: Path) -> str:
        """Ruta relativa a la carpeta del proyecto (también con ../ si está en una carpeta
        vecina); absoluta solo si está en otra unidad. Sin resolve(): en OneDrive cada
        resolve() toca el disco y esto se llama en cada cambio (¿hay algo sin guardar?)."""
        absolute = os.path.abspath(path)
        if self.folder:
            try:
                return Path(os.path.relpath(absolute, os.path.abspath(self.folder))).as_posix()
            except ValueError:                        # otra unidad (Windows)
                pass
        return Path(absolute).as_posix()

    def to_dict(self) -> dict:
        return {
            "fenotit_version": __version__,
            "name": self.name,
            "saved_in": Path(os.path.abspath(self.folder)).as_posix() if self.folder else "",
            "images": [self._rel(path) for path in self.images],
            "current_index": self.current_index,
            "mode": self.mode,
            "analysis": self.analysis,
            "segmentation": asdict(self.segmentation),
            "scale": asdict(self.scale),
            "unit": self.unit,
            "params": self.params,
            "roi": self.roi,
            "image_roi": {self._rel(Path(photo)): value for photo, value in self.image_roi.items()},
            "image_scale": {self._rel(Path(photo)): asdict(value) for photo, value in self.image_scale.items()},
            "excluded_objects": {self._rel(Path(photo)): value for photo, value in self.excluded.items()},
            "highlighted_objects": {self._rel(Path(photo)): value for photo, value in self.highlighted.items()},
            "strokes": {self._rel(Path(photo)): value for photo, value in self.strokes.items() if value},
            "prompts": {self._rel(Path(photo)): value for photo, value in self.prompts.items() if value},
            "regions": {self._rel(Path(photo)): value for photo, value in self.regions.items() if value},
            "corrections": self.corrections.to_dict(),
            "ran": self.ran,
            "preset": self.preset,
            "max_mpx": self.max_mpx,
            "display": self.display,
            "metadata": ({**self.metadata, "file": self._rel(Path(self.metadata["file"]))}
                         if self.metadata.get("file") else {}),
        }

    def save(self, folder: Path | str | None = None) -> Path:
        if folder is not None:
            self.folder = Path(folder)
        if self.folder is None:
            raise ValueError("El proyecto no tiene carpeta")
        self.folder.mkdir(parents=True, exist_ok=True)
        for sub in ("calibration", "results"):        # versiones anteriores las creaban vacías
            sub_folder = self.folder / sub
            try:
                if sub_folder.is_dir() and not any(sub_folder.iterdir()):
                    sub_folder.rmdir()
            except OSError:
                pass
        self.corrections.save_files(self.folder)
        with open(self.file, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, allow_unicode=True, sort_keys=False)
        for old_ext in OLD_PROJECT_EXTS:
            self.file.with_suffix(old_ext).unlink(missing_ok=True)
        return self.file

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        path = Path(path)
        if path.is_dir():
            found = [file for ext in (PROJECT_EXT, *OLD_PROJECT_EXTS, ".yaml") for file in sorted(
                path.glob("project.yaml" if ext == ".yaml" else f"*{ext}"))]
            if not found:
                raise FileNotFoundError(f"No hay un proyecto {PROJECT_EXT} en {path}")
            path = found[0]
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        folder = path.parent
        saved_in = Path(data["saved_in"]) if data.get("saved_in") else None

        def resolve(stored: str) -> Path:
            """Relativa a la carpeta actual del proyecto; si ahí no está y el proyecto se movió
            solo, relativa a la carpeta donde se guardó."""
            candidate = Path(stored)
            if candidate.is_absolute():
                return candidate
            here = Path(os.path.normpath(folder / candidate))
            if saved_in is not None and saved_in != folder and not here.exists():
                there = Path(os.path.normpath(saved_in / candidate))
                if there.exists():
                    return there
            return here

        return cls(
            name=data.get("name", folder.name),
            folder=folder,
            images=[resolve(stored) for stored in data.get("images", [])],
            current_index=int(data.get("current_index", 0)),
            mode=data.get("mode", "individual"),
            analysis=data.get("analysis"),
            segmentation=Segmentation.from_dict(data.get("segmentation")),
            scale=Scale(**data.get("scale", {})),
            unit=data.get("unit", "mm"),
            params=data.get("params", {}) or {},
            roi=from_legacy(data.get("roi")),
            image_roi={cls.key(resolve(photo)): from_legacy(value) for photo, value in (data.get("image_roi") or {}).items()},
            image_scale={cls.key(resolve(photo)): Scale(**value) for photo, value in (data.get("image_scale") or {}).items()},
            excluded={cls.key(resolve(photo)): value for photo, value in (data.get("excluded_objects") or {}).items()},
            highlighted={cls.key(resolve(photo)): value for photo, value in (data.get("highlighted_objects") or {}).items()},
            strokes={cls.key(resolve(photo)): value for photo, value in (data.get("strokes") or {}).items()},
            prompts={cls.key(resolve(photo)): value for photo, value in (data.get("prompts") or {}).items()},
            regions={cls.key(resolve(photo)): value for photo, value in (data.get("regions") or {}).items()},
            corrections=Corrections.from_dict(data.get("corrections"), folder),
            ran=list(data.get("ran") or []),
            preset=dict(data.get("preset") or {}),
            max_mpx=data.get("max_mpx", 50.0),
            display={"legend": True, "color_format": "RGB", "legend_scale": 1.0, **(data.get("display") or {})},
            metadata=({**data["metadata"], "file": str(resolve(data["metadata"]["file"]))}
                      if (data.get("metadata") or {}).get("file") else {}),
        )

    def missing_images(self) -> list[Path]:
        return [path for path in self.images if not path.exists()]

    def relink(self, found: dict) -> int:
        """Cambia la ruta de las fotos movidas (vieja -> nueva) y de todo lo que se guarda por
        foto (áreas, escala, marcas). Devuelve cuántas se cambiaron."""
        old_new = {self.key(old): Path(new) for old, new in found.items()}
        self.images = [old_new.get(self.key(path), path) for path in self.images]
        for name in ("image_roi", "image_scale", "excluded", "highlighted", "strokes", "prompts", "regions"):
            per_photo = getattr(self, name)
            setattr(self, name, {(self.key(old_new[photo]) if photo in old_new else photo): value for photo, value in per_photo.items()})
        return len(old_new)


def find_by_name(missing: list, folder) -> dict:
    """Busca en `folder` (y sus subcarpetas) cada foto que falta por su nombre de archivo, sin
    distinguir mayúsculas. Si hay varias con el mismo nombre, elige la que comparte más
    carpetas finales con la ruta original (p. ej. .../ensayo1/rep2/foto.jpg)."""
    import os
    wanted: dict[str, list[Path]] = {}
    for path in missing:
        wanted.setdefault(Path(path).name.lower(), []).append(Path(path))
    candidates: dict[str, list[Path]] = {}
    for root, _dirs, files in os.walk(folder):
        for file_name in files:
            lowered = file_name.lower()
            if lowered in wanted:
                candidates.setdefault(lowered, []).append(Path(root) / file_name)

    def shared_tail(path_a: Path, path_b: Path) -> int:
        shared = 0
        for x, y in zip(reversed(path_a.parent.parts), reversed(path_b.parent.parts)):
            if x.lower() != y.lower():
                break
            shared += 1
        return shared
    found = {}
    for name, olds in wanted.items():
        cands = list(candidates.get(name, []))
        for old in olds:
            if not cands:
                break
            best = max(cands, key=lambda candidate: shared_tail(old, candidate))
            found[old] = best
            cands.remove(best)
    return found

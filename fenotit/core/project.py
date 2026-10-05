"""Estado de una sesión de trabajo, guardable como carpeta con <carpeta>.fenotit (YAML)."""
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
PROJECT_EXT = ".fenotit"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Segmentation:
    color_space: str = "BGR"
    channel: int = 0
    min_val: int = 0
    max_val: int = 255
    auto: bool = False            # umbral automático (Otsu) en el canal elegido


@dataclass
class Scale:
    mm_per_pixel: float | None = None
    source: str = "none"          # none | manual | two_points | aruco
    detail: str = ""


@dataclass
class Project:
    name: str = "Sin título"
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
    corrections: Corrections = field(default_factory=Corrections)
    metadata: dict[str, str] = field(default_factory=dict)   # {"file": ruta, "key_column": col}
    max_mpx: float | None = 50.0                             # resolución de trabajo del análisis (None = completa)
    ran: list[str] = field(default_factory=list)             # análisis ya corridos (para ofrecer recalcular al abrir)
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
        return sum(1 for k in self.image_roi if k != self.key(path))

    def excluded_for(self, path) -> list[list[float]]:
        return list(self.excluded.get(self.key(path), [])) if path else []

    def set_excluded(self, path, points: list):
        if points:
            self.excluded[self.key(path)] = [[round(float(x), 6), round(float(y), 6)] for x, y in points]
        else:
            self.excluded.pop(self.key(path), None)

    def scale_for(self, path) -> Scale:
        return self.image_scale.get(self.key(path), self.scale) if path else self.scale

    def set_scale(self, path, scale: Scale):
        self.image_scale[self.key(path)] = scale

    def apply_scale_to_all(self, scale: Scale):
        self.scale = scale
        self.image_scale = {}

    def others_with_own_scale(self, path) -> int:
        return sum(1 for k in self.image_scale if k != self.key(path))

    @property
    def preview(self) -> Path | None:
        """Miniatura del último resultado (la muestran los recientes de la pantalla de inicio)."""
        return self.folder / PREVIEW if self.folder else None

    @property
    def file(self) -> Path | None:
        return self.folder / f"{self.folder.name}{PROJECT_EXT}" if self.folder else None

    def set_images(self, paths, mode: str | None = None):
        self.images = [Path(p) for p in paths]
        self.current_index = 0
        if mode:
            self.mode = mode

    def _rel(self, p: Path) -> str:
        """Ruta relativa a la carpeta del proyecto si está dentro. Sin resolve(): en OneDrive
        cada resolve() toca el disco y esto se llama en cada cambio (¿hay algo sin guardar?)."""
        ap = Path(os.path.abspath(p))
        if self.folder:
            try:
                return ap.relative_to(os.path.abspath(self.folder)).as_posix()
            except ValueError:
                pass
        return ap.as_posix()

    def to_dict(self) -> dict:
        return {
            "fenotit_version": __version__,
            "name": self.name,
            "images": [self._rel(p) for p in self.images],
            "current_index": self.current_index,
            "mode": self.mode,
            "analysis": self.analysis,
            "segmentation": asdict(self.segmentation),
            "scale": asdict(self.scale),
            "unit": self.unit,
            "params": self.params,
            "roi": self.roi,
            "image_roi": {self._rel(Path(k)): v for k, v in self.image_roi.items()},
            "image_scale": {self._rel(Path(k)): asdict(v) for k, v in self.image_scale.items()},
            "excluded_objects": {self._rel(Path(k)): v for k, v in self.excluded.items()},
            "corrections": self.corrections.to_dict(),
            "ran": self.ran,
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
        for sub in ("calibration", "results"):
            (self.folder / sub).mkdir(exist_ok=True)
        self.corrections.save_files(self.folder)
        with open(self.file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        return self.file

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        path = Path(path)
        if path.is_dir():
            found = sorted(path.glob(f"*{PROJECT_EXT}")) or sorted(path.glob("project.yaml"))
            if not found:
                raise FileNotFoundError(f"No hay un proyecto {PROJECT_EXT} en {path}")
            path = found[0]
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        folder = path.parent

        def resolve(s: str) -> Path:
            p = Path(s)
            return p if p.is_absolute() else folder / p

        return cls(
            name=d.get("name", folder.name),
            folder=folder,
            images=[resolve(s) for s in d.get("images", [])],
            current_index=int(d.get("current_index", 0)),
            mode=d.get("mode", "individual"),
            analysis=d.get("analysis"),
            segmentation=Segmentation(**d.get("segmentation", {})),
            scale=Scale(**d.get("scale", {})),
            unit=d.get("unit", "mm"),
            params=d.get("params", {}) or {},
            roi=from_legacy(d.get("roi")),
            image_roi={cls.key(resolve(k)): from_legacy(v) for k, v in (d.get("image_roi") or {}).items()},
            image_scale={cls.key(resolve(k)): Scale(**v) for k, v in (d.get("image_scale") or {}).items()},
            excluded={cls.key(resolve(k)): v for k, v in (d.get("excluded_objects") or {}).items()},
            corrections=Corrections.from_dict(d.get("corrections"), folder),
            ran=list(d.get("ran") or []),
            max_mpx=d.get("max_mpx", 50.0),
            display={"legend": True, "color_format": "RGB", "legend_scale": 1.0, **(d.get("display") or {})},
            metadata=({**d["metadata"], "file": str(resolve(d["metadata"]["file"]))}
                      if (d.get("metadata") or {}).get("file") else {}),
        )

    def missing_images(self) -> list[Path]:
        return [p for p in self.images if not p.exists()]

"""Estado de una sesión de trabajo, guardable como carpeta con <carpeta>.fenotit (YAML)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fenotit import __version__
from fenotit.core.corrections.pipeline import Corrections

PROJECT_EXT = ".fenotit"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Segmentation:
    color_space: str = "BGR"
    channel: int = 0
    min_val: int = 0
    max_val: int = 255


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
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    roi: dict[str, list] = field(default_factory=dict)   # formas normalizadas 0-1
    corrections: Corrections = field(default_factory=Corrections)
    metadata: dict[str, str] = field(default_factory=dict)   # {"file": ruta, "key_column": col}

    @property
    def current_image(self) -> Path | None:
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]
        return None

    @property
    def file(self) -> Path | None:
        return self.folder / f"{self.folder.name}{PROJECT_EXT}" if self.folder else None

    def set_images(self, paths, mode: str | None = None):
        self.images = [Path(p) for p in paths]
        self.current_index = 0
        if mode:
            self.mode = mode

    def _rel(self, p: Path) -> str:
        if self.folder:
            try:
                return p.resolve().relative_to(self.folder.resolve()).as_posix()
            except ValueError:
                pass
        return p.resolve().as_posix()

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
            "params": self.params,
            "roi": self.roi,
            "corrections": self.corrections.to_dict(),
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
            params=d.get("params", {}) or {},
            roi=d.get("roi", {}) or {},
            corrections=Corrections.from_dict(d.get("corrections"), folder),
            metadata=({**d["metadata"], "file": str(resolve(d["metadata"]["file"]))}
                      if (d.get("metadata") or {}).get("file") else {}),
        )

    def missing_images(self) -> list[Path]:
        return [p for p in self.images if not p.exists()]

"""Cadena de análisis: segmentadores → procesadores → mediciones.

Cada pieza se registra con @step y declara qué necesita (requires) y qué produce
(provides) del contexto. Una cadena es una lista serializable:
    [{"step": "threshold", "params": {...}}, {"step": "morphometry"}, ...]
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from fenotit import log

_log = log.get("pipeline")

KINDS = ("segmenter", "processor", "measurement")


@dataclass
class Context:
    image: np.ndarray
    mm_per_px: float | None = None
    roi: np.ndarray | None = None                    # uint8, tamaño de la imagen
    exclusions: list = field(default_factory=list)   # [(puntos Nx2, color BGR)]
    mask: np.ndarray | None = None                   # uint8 0/255
    labels: np.ndarray | None = None                 # int32, 0 = fondo
    detections: list[dict] | None = None             # [{"x", "y", ...}]
    skeleton: np.ndarray | None = None
    groups: np.ndarray | None = None                 # componentes antes de desagrupar
    tables: dict[str, list[dict]] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def unit(self) -> str:
        return "mm" if self.mm_per_px else "px"

    @property
    def scale(self) -> float:
        return self.mm_per_px or 1.0

    def object_rows(self) -> dict[int, dict]:
        """Filas de la tabla 'objects' indexadas por id de objeto (se crean si faltan)."""
        rows = self.extra.setdefault("_object_rows", {})
        if self.labels is not None and not rows:
            for oid in np.unique(self.labels):
                if oid:
                    rows[int(oid)] = {"object_id": int(oid)}
            self.tables["objects"] = list(rows.values())
        return rows

    def touching_ids(self) -> set[int]:
        """Objetos que salieron de desagrupar un grupo (se tocaban con otro)."""
        if self.groups is None or self.labels is None:
            return set()
        fg = self.labels > 0
        pairs = np.unique(np.stack([self.labels[fg], self.groups[fg]]), axis=1)
        gid, n = np.unique(pairs[1], return_counts=True)
        shared = set(gid[n > 1].tolist())
        return {int(o) for o, g in pairs.T if g in shared}

    def image_row(self) -> dict:
        t = self.tables.setdefault("image", [{}])
        return t[0]


@dataclass
class Step:
    key: str
    kind: str
    func: Callable[[Context, dict], None]
    params: list[dict] = field(default_factory=list)
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    label: str = ""

    def defaults(self) -> dict:
        return {p["key"]: p["default"] for p in self.params}


REGISTRY: dict[str, Step] = {}


class PipelineError(RuntimeError):
    pass


def step(key: str, kind: str, params: list[dict] | None = None,
         requires: tuple[str, ...] = (), provides: tuple[str, ...] = (), label: str = ""):
    assert kind in KINDS, kind

    def deco(func):
        REGISTRY[key] = Step(key, kind, func, params or [], tuple(requires), tuple(provides), label or key)
        return func
    return deco


def discover() -> dict[str, Step]:
    """Importa todas las piezas de fenotit.core.pipeline.steps."""
    from fenotit.core.pipeline import steps
    for mod in pkgutil.iter_modules(steps.__path__):
        try:
            importlib.import_module(f"{steps.__name__}.{mod.name}")
        except Exception:
            _log.exception("No se pudo cargar la pieza %s", mod.name)
    return REGISTRY


def run(image: np.ndarray, chain: list[dict], mm_per_px: float | None = None,
        roi: np.ndarray | None = None, exclusions: list | None = None) -> Context:
    if any(item["step"] not in REGISTRY for item in chain):
        discover()
    ctx = Context(image=image, mm_per_px=mm_per_px, roi=roi, exclusions=exclusions or [])
    for item in chain:
        s = REGISTRY.get(item["step"])
        if s is None:
            raise PipelineError(f"unknown step '{item['step']}'")
        missing = [r for r in s.requires if getattr(ctx, r) is None]
        if missing:
            raise PipelineError(f"'{s.key}' needs {missing}")
        params = {**s.defaults(), **(item.get("params") or {})}
        s.func(ctx, params)
    return ctx

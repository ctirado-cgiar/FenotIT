"""Recalcular solo lo que cambió.

La cadena es: segmentación y objetos (prefijo) → mediciones. Las mediciones solo leen
los objetos y no dependen entre sí. Se guarda el resultado del prefijo y el aporte de
cada medición; si al volver a correr solo cambian los colores, no se vuelve a
segmentar ni a medir morfometría.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

import numpy as np

# El registro de piezas y el contexto que la caché guarda y copia entre ejecuciones.
from fenotit.core.pipeline.base import REGISTRY, Context, PipelineError, discover


def _hash(*parts) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str(part.shape).encode())
            digest.update(np.ascontiguousarray(part[::7, ::7] if part.ndim >= 2 else part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()


class ChainCache:
    """Caché de las últimas `size` imágenes (en memoria)."""

    def __init__(self, size: int = 4):
        self.size = size
        self._prefix: OrderedDict[str, Context] = OrderedDict()
        self._measures: OrderedDict[str, dict] = OrderedDict()
        self.hits: list[str] = []          # piezas reutilizadas en la última corrida

    def _remember(self, store: OrderedDict, key: str, value, limit: int):
        store[key] = value
        store.move_to_end(key)
        while len(store) > limit:
            store.popitem(last=False)

    def run(self, image: np.ndarray, chain: list[dict], mm_per_px: float | None = None,
            roi: np.ndarray | None = None, exclusions: list | None = None, unit: str = "mm") -> Context:
        if any(item["step"] not in REGISTRY for item in chain):
            discover()
        kinds = [REGISTRY[item["step"]].kind if item["step"] in REGISTRY else None for item in chain]
        if None in kinds:
            raise PipelineError(f"unknown step in {[i['step'] for i in chain]}")
        split = next((i for i, step_kind in enumerate(kinds) if step_kind == "measurement"), len(chain))
        if any(step_kind != "measurement" for step_kind in kinds[split:]):          # orden no estándar
            from fenotit.core.pipeline.base import run
            return run(image, chain, mm_per_px, roi, exclusions, unit)

        exclusions = exclusions or []
        base = _hash(image, roi if roi is not None else "", [(np.asarray(points).tolist(), exclusion_kind) for points, exclusion_kind in exclusions])
        self.hits = []

        pkey = _hash(base, chain[:split])
        prefix = self._prefix.get(pkey)
        if prefix is None:
            prefix = Context(image=image, mm_per_px=mm_per_px, length_unit=unit, roi=roi, exclusions=exclusions)
            for item in chain[:split]:
                _apply(prefix, item)
            self._remember(self._prefix, pkey, prefix, self.size)
        else:
            self.hits.append("objects")

        deltas = []
        for item in chain[split:]:
            mkey = _hash(pkey, item, mm_per_px, unit)
            delta = self._measures.get(mkey)
            if delta is None:
                delta = _measure(prefix, item, mm_per_px, unit)
                self._remember(self._measures, mkey, delta, self.size * 8)
            else:
                self.hits.append(item["step"])
            deltas.append(delta)
        return _compose(prefix, deltas, mm_per_px, unit)


def _apply(ctx: Context, item: dict):
    piece = REGISTRY[item["step"]]
    missing = [requirement for requirement in piece.requires if getattr(ctx, requirement) is None]
    if missing:
        raise PipelineError(f"'{piece.key}' needs {missing}")
    piece.func(ctx, {**piece.defaults(), **(item.get("params") or {})})


def _fresh(prefix: Context, mm_per_px, unit="mm") -> Context:
    return Context(image=prefix.image, mm_per_px=mm_per_px, length_unit=unit, roi=prefix.roi, exclusions=prefix.exclusions,
                   mask=prefix.mask, labels=prefix.labels, detections=prefix.detections,
                   skeleton=prefix.skeleton, groups=prefix.groups, classes=prefix.classes,
                   excluded=prefix.excluded,
                   highlighted=prefix.highlighted,
                   extra={key: prefix.extra[key] for key in ("class_names", "class_colors") if key in prefix.extra})


def _measure(prefix: Context, item: dict, mm_per_px, unit="mm") -> dict:
    """Corre una medición sola y guarda lo que aporta."""
    ctx = _fresh(prefix, mm_per_px, unit)
    _apply(ctx, item)
    rows = {row["object_id"]: {name: table for name, table in row.items() if name != "object_id"}
            for row in ctx.tables.get("objects", [])}
    image_row = dict(ctx.tables["image"][0]) if ctx.tables.get("image") else {}
    others = {name: table for name, table in ctx.tables.items() if name not in ("objects", "image")}
    new = {name: table for name, table in ctx.extra.items() if name != "overlays" and name not in prefix.extra}
    return {"rows": rows, "image": image_row, "tables": others, "images": dict(ctx.images),
            "overlays": dict(ctx.extra.get("overlays", {})), "extra": new}


def _compose(prefix: Context, deltas: list[dict], mm_per_px, unit="mm") -> Context:
    ctx = _fresh(prefix, mm_per_px, unit)
    ctx.images = dict(prefix.images)
    ctx.extra = {key: value for key, value in prefix.extra.items() if not key.startswith("_")}
    if any(delta["rows"] for delta in deltas):
        rows = ctx.object_rows()
        for delta in deltas:
            for oid, cols in delta["rows"].items():
                rows[oid].update(cols)
    for delta in deltas:
        if delta["image"]:
            ctx.image_row().update(delta["image"])
        ctx.tables.update(delta["tables"])
        ctx.images.update(delta["images"])
        if delta.get("overlays"):
            ctx.extra.setdefault("overlays", {}).update(delta["overlays"])
        ctx.extra.update(delta.get("extra") or {})
    return ctx

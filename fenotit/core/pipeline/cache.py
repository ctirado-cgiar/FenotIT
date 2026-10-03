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

from fenotit.core.pipeline.base import REGISTRY, Context, PipelineError, discover


def _hash(*parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(np.ascontiguousarray(p[::7, ::7] if p.ndim >= 2 else p).tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


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
        split = next((i for i, k in enumerate(kinds) if k == "measurement"), len(chain))
        if any(k != "measurement" for k in kinds[split:]):          # orden no estándar
            from fenotit.core.pipeline.base import run
            return run(image, chain, mm_per_px, roi, exclusions, unit)

        exclusions = exclusions or []
        base = _hash(image, roi if roi is not None else "", [(np.asarray(p).tolist(), c) for p, c in exclusions])
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
    s = REGISTRY[item["step"]]
    missing = [r for r in s.requires if getattr(ctx, r) is None]
    if missing:
        raise PipelineError(f"'{s.key}' needs {missing}")
    s.func(ctx, {**s.defaults(), **(item.get("params") or {})})


def _fresh(prefix: Context, mm_per_px, unit="mm") -> Context:
    return Context(image=prefix.image, mm_per_px=mm_per_px, length_unit=unit, roi=prefix.roi, exclusions=prefix.exclusions,
                   mask=prefix.mask, labels=prefix.labels, detections=prefix.detections,
                   skeleton=prefix.skeleton, groups=prefix.groups)


def _measure(prefix: Context, item: dict, mm_per_px, unit="mm") -> dict:
    """Corre una medición sola y guarda lo que aporta."""
    ctx = _fresh(prefix, mm_per_px, unit)
    _apply(ctx, item)
    rows = {r["object_id"]: {k: v for k, v in r.items() if k != "object_id"}
            for r in ctx.tables.get("objects", [])}
    image_row = dict(ctx.tables["image"][0]) if ctx.tables.get("image") else {}
    others = {k: v for k, v in ctx.tables.items() if k not in ("objects", "image")}
    return {"rows": rows, "image": image_row, "tables": others, "images": dict(ctx.images),
            "overlays": dict(ctx.extra.get("overlays", {}))}


def _compose(prefix: Context, deltas: list[dict], mm_per_px, unit="mm") -> Context:
    ctx = _fresh(prefix, mm_per_px, unit)
    ctx.images = dict(prefix.images)
    ctx.extra = {k: v for k, v in prefix.extra.items() if not k.startswith("_")}
    if any(d["rows"] for d in deltas):
        rows = ctx.object_rows()
        for d in deltas:
            for oid, cols in d["rows"].items():
                rows[oid].update(cols)
    for d in deltas:
        if d["image"]:
            ctx.image_row().update(d["image"])
        ctx.tables.update(d["tables"])
        ctx.images.update(d["images"])
        if d.get("overlays"):
            ctx.extra.setdefault("overlays", {}).update(d["overlays"])
    return ctx

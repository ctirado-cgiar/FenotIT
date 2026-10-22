"""Segmentación como paso aparte: separa lo que interesa (objetos, clases) del fondo.

Métodos:
  threshold   umbral en un canal de color (rango Mín-Máx u Otsu)
  classifier  por ejemplos: el usuario pinta trazos de cada clase y un bosque aleatorio
              clasifica cada píxel (como ilastik); da una máscara y un mapa de clases
  model       un modelo de IA (MobileSAM u otro en ONNX)

Después, "Afinar máscara": una lista de operaciones morfológicas en orden.

La configuración es un dict serializable (va al proyecto, a los presets y a los
metadatos de la exportación); `chain(seg)` la vuelve piezas del pipeline."""
from __future__ import annotations

import cv2
import numpy as np

METHODS = ("threshold", "classifier", "model")

# operaciones de "Afinar máscara" (tamaño en px del elemento circular; "veces" = iteraciones)
REFINE_OPS = {
    "erode": {"size": True},          # adelgaza: quita puntos y uniones finas
    "dilate": {"size": True},         # engrosa: une partes cercanas
    "open": {"size": True},           # erosión + dilatación: quita ruido sin encoger lo grande
    "close": {"size": True},          # dilatación + erosión: cierra grietas y huecos pequeños
    "open_rec": {"size": True},       # apertura por reconstrucción: quita lo pequeño sin tocar
                                      # el borde de lo que queda
    "fill_holes": {"size": False},    # rellena huecos dentro de los objetos
    "median": {"size": True},         # suaviza el borde (filtro de mediana)
}

DEFAULT_REFINE = [{"op": "fill_holes"}]


def _kernel(size: int):
    size = max(1, int(size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def refine_op(mask: np.ndarray, op: str, size: int = 3, iterations: int = 1) -> np.ndarray:
    it = max(1, int(iterations))
    if op in ("erode", "dilate", "open", "close"):
        code = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
                "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}[op]
        return cv2.morphologyEx(mask, code, _kernel(size), iterations=it)
    if op == "open_rec":
        seed = cv2.erode(mask, _kernel(size), iterations=it)
        return reconstruct(seed, mask)
    if op == "fill_holes":
        from fenotit.core.pipeline.steps.objects import _fill_small_holes
        return _fill_small_holes(mask)
    if op == "median":
        k = max(3, int(size) | 1)
        out = mask
        for _ in range(it):
            out = cv2.medianBlur(out, min(k, 255))
        return out
    return mask


def reconstruct(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstrucción morfológica por dilatación, rápida: se quedan las componentes de la
    máscara que tocan alguna semilla (equivale a dilatar hasta estabilizar)."""
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[seed > 0])] = True
    keep[0] = False
    return keep[lab].astype(np.uint8) * 255


def refine(mask: np.ndarray, ops: list[dict] | None, scale: float = 1.0) -> np.ndarray:
    """Aplica las operaciones en orden. `scale` < 1 en la vista previa reducida."""
    for o in ops or []:
        op = o.get("op")
        if op not in REFINE_OPS:
            continue
        size = max(1, round(int(o.get("size", 3)) * scale)) if REFINE_OPS[op]["size"] else 0
        mask = refine_op(mask, op, size, int(o.get("iter", 1)))
    return mask


def from_legacy(params: dict) -> dict:
    """Parámetros planos de versiones anteriores (color_space, channel_idx, min_val…)."""
    seg = {"method": "threshold", "color_space": params.get("color_space", "BGR"),
           "channel": int(params.get("channel_idx", 0)), "min_val": int(params.get("min_val", 0)),
           "max_val": int(params.get("max_val", 255)), "auto": bool(params.get("auto_threshold"))}
    if params.get("morph_op", "none") != "none":
        seg["refine"] = [{"op": params["morph_op"], "size": int(params.get("morph_size", 3)),
                          "iter": int(params.get("morph_iter", 1))}] + DEFAULT_REFINE
    return seg


def chain(seg: dict) -> list[dict]:
    """Piezas de segmentación (sin roi): segmentador y, si hay, afinar máscara."""
    method = seg.get("method", "threshold")
    if method == "classifier":
        c = seg.get("classifier") or {}
        steps = [{"step": "classify", "params": {"samples": c.get("samples", ""), "hash": c.get("hash", ""),
                                                 "objects": list(c.get("objects", [1])),
                                                 "colors": list(c.get("colors", [])),
                                                 "detail": float(c.get("detail", 1.0))}}]
    elif method == "model":
        m = seg.get("model") or {}
        steps = [{"step": "model_segment", "params": {k: v for k, v in m.items()}}]
    elif seg.get("auto"):
        steps = [{"step": "otsu", "params": {"color_space": seg.get("color_space", "BGR"),
                                             "channel": int(seg.get("channel", 0))}}]
    else:
        steps = [{"step": "threshold", "params": {"color_space": seg.get("color_space", "BGR"),
                                                  "channel": int(seg.get("channel", 0)),
                                                  "min_val": int(seg.get("min_val", 0)),
                                                  "max_val": int(seg.get("max_val", 255))}}]
    return steps


def refine_steps(seg: dict) -> list[dict]:
    if "refine" not in seg:
        return []
    return [{"step": "refine", "params": {"ops": list(seg.get("refine") or [])}}]

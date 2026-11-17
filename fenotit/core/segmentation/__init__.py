"""Segmentación como paso aparte: separa lo que interesa (objetos, clases) del fondo.

Métodos:
  threshold   umbral en un canal de color (rango Mín-Máx u Otsu)
  classifier  por ejemplos: el usuario pinta trazos de cada clase y un bosque aleatorio
              clasifica cada píxel (como ilastik); da una máscara y un mapa de clases
  model       un modelo de IA en ONNX (core/segmentation/models.py): de máscaras (MobileSAM)
              o de cajas (detector); las cajas pueden pasar a máscaras con MobileSAM

Después, "Afinar máscara": una lista de operaciones morfológicas en orden.

La configuración es un dict serializable (va al proyecto, a los presets y a los
metadatos de la exportación); `chain(seg)` la vuelve piezas del pipeline."""
from __future__ import annotations


METHODS = ("threshold", "classifier", "model")

from fenotit.core.segmentation.refine_mask import DEFAULT_REFINE, REFINE_OPS, reconstruct, refine, refine_op

__all__ = ["DEFAULT_REFINE", "METHODS", "REFINE_OPS", "boxes_only", "chain", "from_legacy", "reconstruct",
           "refine", "refine_op", "refine_steps"]


def boxes_only(seg: dict) -> bool:
    """La segmentación da cajas (detector sin pasar las cajas a máscaras)."""
    model = seg.get("model") or {}
    return seg.get("method") == "model" and model.get("output") == "boxes" and not model.get("to_masks")


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
        classifier_cfg = seg.get("classifier") or {}
        steps = [{"step": "classify", "params": {"samples": classifier_cfg.get("samples", ""), "hash": classifier_cfg.get("hash", ""),
                                                 "objects": list(classifier_cfg.get("objects", [1])),
                                                 "colors": list(classifier_cfg.get("colors", [])),
                                                 "detail": float(classifier_cfg.get("detail", 1.0))}}]
    elif method == "model" and (seg.get("model") or {}).get("output") == "boxes":
        model = seg["model"]
        steps = [{"step": "detect", "params": {"model": model.get("name", ""), "conf": float(model.get("conf", 0.25)),
                                               "use": [int(classifier_cfg) for classifier_cfg in model.get("use") or []],
                                               "to_masks": bool(model.get("to_masks")),
                                               "colors": list(model.get("colors") or [])}}]
    elif method == "model":
        model = seg.get("model") or {}
        steps = [{"step": "model_segment", "params": {"model": model.get("name", "mobilesam"),
                                                      "mode": model.get("mode", "auto"),
                                                      "grid": int(model.get("grid", 16)),
                                                      "points": list(model.get("points") or []),
                                                      "categories": [classifier_cfg["name"] for classifier_cfg in model.get("categories") or []],
                                                      "colors": [classifier_cfg["color"] for classifier_cfg in model.get("categories") or []],
                                                      "default": int(model.get("default", 1)),
                                                      "picks": list(model.get("picks") or []),
                                                      "own": bool(model.get("own", True)),
                                                      "learn": dict(model.get("learn") or {})}}]
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

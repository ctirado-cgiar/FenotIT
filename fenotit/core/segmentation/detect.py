"""Detección de objetos con un modelo ONNX que da una caja y una clase por objeto, en CPU y
sin internet. Formatos de salida que se leen:

  yolo  una salida [1, 4+clases, N] o [1, N, 4+clases] (YOLOv8/v11, cajas centro-ancho-alto
        en píxeles de la entrada), [1, N, 5+clases] (YOLOv5, con confianza de objeto) o
        [1, N, 6] (salida final x0, y0, x1, y1, puntaje, clase)
  detr  dos salidas: cajas [1, N, 4] (centro-ancho-alto, 0-1) y puntajes [1, N, clases]
        (RF-DETR, DETR)

La foto se lleva al tamaño de entrada del modelo (yolo: sin deformar, con relleno gris;
detr: estirada) y las cajas vuelven a píxeles de la foto. Cajas repetidas de la misma
clase se quitan con supresión de no máximos (NMS)."""
from __future__ import annotations

import ast

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_SESSIONS: dict = {}


def session(path: str):
    if path not in _SESSIONS:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False
        _SESSIONS[path] = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return _SESSIONS[path]


def _meta_names(sess) -> list[str]:
    """Clases guardadas en los metadatos del ONNX (Ultralytics: names = {0: 'a', ...})."""
    raw = sess.get_modelmeta().custom_metadata_map.get("names")
    if not raw:
        return []
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in names] if isinstance(names, (list, tuple)) else []


def _dim(v, default: int) -> int:
    return v if isinstance(v, int) and v > 0 else default


def inspect(path: str) -> dict:
    """Formato, tamaño de entrada, número de clases, nombres y ficha (descripción, autor,
    licencia, documentación) si el archivo los trae.
    ValueError("format") si las salidas no corresponden a un formato conocido."""
    sess = session(path)
    shape = sess.get_inputs()[0].shape
    size = (_dim(shape[2], 0) or _dim(shape[3], 0)) if len(shape) == 4 else 0
    names = _meta_names(sess)
    outs = [o.shape for o in sess.get_outputs()]
    found = sess.get_modelmeta().custom_metadata_map
    info = {"input": int(size or 640), "classes": names,
            "meta": {k: found[k] for k in ("description", "author", "license", "docs", "version", "date") if k in found}}
    if len(outs) == 1 and len(outs[0]) == 3:
        a, b = _dim(outs[0][1], 0), _dim(outs[0][2], 0)
        layout = _yolo_layout(a, b, len(names))
        if layout:
            return {**info, "format": "yolo", "n_classes": layout[1] or len(names)}
    if len(outs) >= 2:
        boxes = next((s for s in outs if len(s) == 3 and s[2] == 4), None)
        logits = next((s for s in outs if len(s) == 3 and s is not boxes), None)
        if boxes is not None and logits is not None:
            nc = _dim(logits[2], 0)
            if names and nc == len(names) + 1:
                nc -= 1                       # índice 0 reservado (exportaciones de Roboflow)
            return {**info, "format": "detr", "n_classes": nc or len(names)}
    raise ValueError("format")


def _yolo_layout(a: int, b: int, n_names: int):
    """(diseño, clases) de una salida yolo [1, a, b]: 'cf' = [1, 4+c, N]; 'nf' = [1, N, 4+c];
    'nf5' = [1, N, 5+c] (YOLOv5); 'final' = [1, N, 6]. None si no encaja."""
    if not a or not b:
        return None
    if n_names:
        if a == 4 + n_names:
            return "cf", n_names
        if b == 4 + n_names:
            return "nf", n_names
        if b == 5 + n_names:
            return "nf5", n_names
    if b == 6 and a <= 1000:
        return "final", 0
    if a < b:
        return ("cf", a - 4) if a > 4 else None
    return ("nf5", b - 5) if b > 5 else None


def _letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    px, py = (size - nw) / 2, (size - nh) / 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[int(round(py - 0.1)):int(round(py - 0.1)) + nh, int(round(px - 0.1)):int(round(px - 0.1)) + nw] = \
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return out, r, int(round(px - 0.1)), int(round(py - 0.1))


def _nms(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou: float) -> list[int]:
    keep = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[idx]]
        got = cv2.dnn.NMSBoxes(xywh, scores[idx].astype(float).tolist(), 0.0, iou)
        keep += [int(idx[i]) for i in np.asarray(got).reshape(-1)]
    return sorted(keep, key=lambda i: -scores[i])


def run(img: np.ndarray, model: dict, conf: float = 0.25, iou: float = 0.5) -> list[dict]:
    """Detecciones en la foto BGR: [{x0, y0, x1, y1 (px de la foto), cls (0..), score}],
    de mayor a menor puntaje."""
    from pathlib import Path
    sess = session(str(Path(model["dir"]) / model["files"][0]))
    size, fmt = int(model.get("input") or 640), model.get("format", "yolo")
    h, w = img.shape[:2]
    rgb_src = cv2.cvtColor(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    name = sess.get_inputs()[0].name
    if fmt == "detr":
        x = cv2.resize(rgb_src, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        outs = sess.run(None, {name: x})
        boxes = next(o for o in outs if o.ndim == 3 and o.shape[2] == 4)[0]
        logits = next(o for o in outs if o.ndim == 3 and o.shape[2] != 4)[0]
        prob = 1 / (1 + np.exp(-logits.astype(np.float32)))
        offset = 1 if model.get("classes") and prob.shape[1] == len(model["classes"]) + 1 else 0
        prob = prob[:, offset:]
        cls, score = prob.argmax(1), prob.max(1)
        cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    else:
        lb, r, px, py = _letterbox(rgb_src, size)
        x = (lb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        out = sess.run(None, {name: x})[0][0]
        layout, _ = _yolo_layout(out.shape[0], out.shape[1], len(model.get("classes") or [])) or ("nf", 0)
        if layout == "cf":
            out = out.T
        if layout == "final":
            xyxy, score, cls = out[:, :4].copy(), out[:, 4], out[:, 5].astype(int)
        else:
            probs = out[:, 4:] if layout in ("cf", "nf") else out[:, 5:] * out[:, 4:5]
            cls, score = probs.argmax(1), probs.max(1)
            cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
    ok = score >= conf
    xyxy, score, cls = xyxy[ok], score[ok], np.asarray(cls)[ok]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    good = (xyxy[:, 2] - xyxy[:, 0] >= 1) & (xyxy[:, 3] - xyxy[:, 1] >= 1)
    xyxy, score, cls = xyxy[good], score[good], cls[good]
    keep = _nms(xyxy, score, cls, iou) if len(score) else []
    return [{"x0": float(xyxy[i, 0]), "y0": float(xyxy[i, 1]), "x1": float(xyxy[i, 2]), "y1": float(xyxy[i, 3]),
             "cls": int(cls[i]), "score": float(score[i])} for i in keep]


def boxes_mask(shape, dets: list[dict]) -> np.ndarray:
    """Máscara (0/255) con las cajas rellenas."""
    m = np.zeros(shape[:2], np.uint8)
    for d in dets:
        cv2.rectangle(m, (int(d["x0"]), int(d["y0"])), (int(np.ceil(d["x1"])) - 1, int(np.ceil(d["y1"])) - 1), 255, -1)
    return m


def boxes_labels(shape, dets: list[dict]) -> np.ndarray:
    """Etiquetas con cada caja rellena (id = posición + 1); las cajas pequeñas quedan encima
    de las grandes."""
    lab = np.zeros(shape[:2], np.int32)
    order = sorted(range(len(dets)), key=lambda i: -(dets[i]["x1"] - dets[i]["x0"]) * (dets[i]["y1"] - dets[i]["y0"]))
    for i in order:
        d = dets[i]
        lab[int(d["y0"]):int(np.ceil(d["y1"])), int(d["x0"]):int(np.ceil(d["x1"]))] = i + 1
    return lab


# ── Pieza del pipeline ──────────────────────────────────────────────────────

from fenotit.core.pipeline.base import step  # noqa: E402


@step("detect", "segmenter", provides=("mask",), params=[
    {"key": "model", "type": "str", "default": ""},
    {"key": "conf", "type": "float", "default": 0.25, "min": 0.01, "max": 1.0},
    {"key": "use", "type": "list", "default": []},
    {"key": "to_masks", "type": "bool", "default": False},
    {"key": "colors", "type": "list", "default": []},
])
def detect(ctx, p):
    """Cajas de un detector. `use` = índices de las clases que se conservan (vacío = todas).
    Sin `to_masks`: las cajas quedan en ctx.detections (cada una con su clase) y la máscara
    es el relleno de las cajas. Con `to_masks`: MobileSAM dibuja el contorno del objeto
    dentro de cada caja; los objetos quedan en ctx.extra["instances"] y su clase en
    ctx.classes (1..)."""
    from fenotit.core.segmentation import models
    model = models.get(p["model"])
    if model is None or model["output"] != "boxes":
        raise FileNotFoundError(f"Detection model not found: {p['model']}")
    dets = run(ctx.image, model, float(p["conf"]))
    use = set(int(c) for c in p.get("use") or [])
    if use:
        dets = [d for d in dets if d["cls"] in use]
    names = list(model["classes"])
    for d in dets:
        d["class"] = names[d["cls"]] if d["cls"] < len(names) else f"class_{d['cls'] + 1}"
    ctx.extra["class_names"] = names
    ctx.extra["class_colors"] = ["#000000"] + list(p.get("colors") or [])
    if p.get("to_masks"):
        from fenotit.core.segmentation import sam
        labels = sam.from_boxes(ctx.image, [(d["x0"], d["y0"], d["x1"], d["y1"]) for d in dets])
        cls = np.zeros(len(dets) + 1, np.int32)
        cls[1:] = [d["cls"] + 1 for d in dets]
        ctx.extra["instances"] = labels
        ctx.classes = cls[labels]
        ctx.mask = (labels > 0).astype(np.uint8) * 255
    else:
        ctx.detections = dets
        ctx.mask = boxes_mask(ctx.image.shape, dets)
    ctx.images["mask"] = ctx.mask
